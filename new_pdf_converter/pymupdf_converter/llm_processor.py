# src/pdf/pymupdf4llm/processor.py (修正版)
import hashlib
from pathlib import Path
from typing import Dict, List, Callable, Optional
import traceback

import pymupdf4llm

from .llm_models import ProcessedDocument, DocumentMetadata, ProcessingSettings
from .llm_chunker import MarkdownChunker
from utils.log_manager import LogManager
from utils.node_threshold_manager import NodeThresholdManager
from utils.key_registry import KeyRegistry
from utils.agents.vlm_executor import VLMExecutor
from vlm.model_manager import ModelManager
from vlm.caption_maker.type_config import DEFAULT_TYPE_MAP, CLASSIFIER_PARAMS, CLASSIFIER_PROMPT

class PyMuPDFProcessor:
    """PyMuPDF4LLM処理器（エラーハンドリング強化版）"""
    
    def __init__(self, settings: ProcessingSettings = None, progress_cb: Optional[Callable[[Dict], None]] = None):
        self.settings = settings or ProcessingSettings()
        self.chunker = MarkdownChunker(
            max_chunk_size=self.settings.chunk_size,
            overlap_size=self.settings.overlap_size,
            rag_emit_page=(self.settings.rag_settings or {}).get("emit_page_number", True)
        )
        self.progress_cb = progress_cb
    
    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        """PDF処理メイン（エラーハンドリング強化）"""
        print(f"🔄 PDF処理開始: {pdf_path}")
        
        try:
            # 1. Markdown変換（プログレス表示付き）
            print("📄 Markdown変換中...")
            markdown_text = self._safe_markdown_conversion(pdf_path)
            
            if not markdown_text or not markdown_text.strip():
                raise ValueError("PDFからテキストを抽出できませんでした")
            
            print(f"✅ Markdown変換完了: {len(markdown_text):,}文字")
            
            # 2. メタデータ作成
            print("📋 メタデータ作成中...")
            doc_metadata = self._create_metadata(pdf_path)

            # 2.5 画像キャプション（write_imagesかつgenerate_captionsが有効な場合）
            try:
                kwargs = self._filter_supported_kwargs(self.settings.pymupdf_kwargs or {})
                write_images_on = bool(kwargs.get("write_images")) and not bool(kwargs.get("embed_images"))
                if getattr(self.settings, 'generate_captions', False) and write_images_on:
                    base_dir = Path(__file__).resolve().parents[1]  # new_pdf_converter
                    images_dir = base_dir / "images"
                    caption_dir = base_dir / "caption"
                    print(f"🖼️ 画像キャプション生成中... ({images_dir})")
                    mapping = self._classify_and_caption_stream(images_dir, caption_dir, markdown_text)
                    if mapping:
                        # Markdown画像タグ全体をキャプションに置換
                        markdown_text = self._replace_markdown_images(markdown_text, mapping)
                        print(f"✅ キャプション置換完了: {len(mapping)}件")
            except Exception as e:
                print(f"⚠️ 画像キャプション生成スキップ: {e}")

            # 3. チャンク分割（安全実行）
            print("✂️ チャンク分割中...")
            chunks = self.chunker.chunk_markdown(markdown_text, doc_metadata)

            if not chunks:
                print("⚠️ チャンクが生成されませんでした。フォールバック実行中...")
                chunks = self.chunker._create_fallback_chunk(markdown_text, doc_metadata)

            # 3.5 RAGメタ設定の反映（軽量）
            self._apply_rag_metadata(chunks, doc_metadata)
            
            # 4. 統計計算（安全版）
            print("📊 統計計算中...")
            stats = self._calculate_stats_safe(markdown_text, chunks)
            
            result = ProcessedDocument(
                document_metadata=doc_metadata,
                chunks=chunks,
                raw_markdown=markdown_text,
                processing_stats=stats
            )
            
            print(f"✅ PDF処理完了: {len(chunks)}個のチャンク生成")
            return result
            
        except Exception as e:
            print(f"❌ PDF処理エラー: {e}")
            print(f"📍 エラー詳細:\n{traceback.format_exc()}")
            
            # エラー時のフォールバック処理
            return self._create_error_fallback(pdf_path, str(e))
    
    def _safe_markdown_conversion(self, pdf_path: str) -> str:
        """安全なMarkdown変換"""
        try:
            # バージョン情報（デバッグ）
            try:
                import importlib.metadata as _md
                _ver_pymupdf4llm = _md.version("pymupdf4llm")
            except Exception:
                _ver_pymupdf4llm = "unknown"
            try:
                import fitz as _fz
                _ver_pymupdf = getattr(_fz, "__doc__", "").split("PyMuPDF")[-1].strip() or "unknown"
            except Exception:
                _ver_pymupdf = "unknown"
            print(f"🧩 pymupdf4llm={_ver_pymupdf4llm}, pymupdf={_ver_pymupdf}")

            raw_kwargs = dict(self.settings.pymupdf_kwargs or {})
            print("🔧 PyMuPDF4LLM kwargs (raw):", raw_kwargs)

            # スキーマベースのkwargsを適用（未対応キーは除外）
            kwargs = self._filter_supported_kwargs(raw_kwargs)
            print("✅ PyMuPDF4LLM kwargs (effective):", kwargs)

            result = pymupdf4llm.to_markdown(pdf_path, **kwargs)
            # デバッグ: page_chunksの有効性
            requested_page_chunks = bool(kwargs.get("page_chunks"))
            if isinstance(result, list):
                print(f"🧪 to_markdown returned list (pages) — count={len(result)}; page_chunks requested={requested_page_chunks}")
                # 追加デバッグ: ページ配列の要素構造を確認
                self._debug_inspect_page_chunks(result)
            else:
                print(f"🧪 to_markdown returned str — length={len(result) if isinstance(result, str) else 'n/a'}; page_chunks requested={requested_page_chunks}")
                if requested_page_chunks:
                    print("⚠️ page_chunks=True が要求されましたが、戻り値は文字列でした。ライブラリの仕様/バージョン差異の可能性。")
            # page_chunks=True の場合はリストが返る可能性があるので統一
            return self._normalize_markdown_result(result)
            
        except Exception as first_error:
            print(f"⚠️ 標準変換失敗: {first_error}")
            
            try:
                # ページ指定で少しずつ処理
                print("🔄 ページ別変換を試行中...")
                import fitz
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                
                markdown_parts = []
                
                for page_num in range(total_pages):
                    try:
                        print(f"📄 ページ {page_num + 1}/{total_pages} 処理中...")
                        kwargs = self._filter_supported_kwargs(self.settings.pymupdf_kwargs or {})
                        # 個別ページ処理
                        page_md = pymupdf4llm.to_markdown(pdf_path, pages=[page_num], **kwargs)
                        if isinstance(page_md, list):
                            print(f"🧪 page {page_num+1}: list returned (len={len(page_md)})")
                            self._debug_inspect_page_chunks(page_md, label=f"page {page_num+1}")
                        elif isinstance(page_md, str):
                            print(f"🧪 page {page_num+1}: str returned (len={len(page_md)})")
                        page_text = self._normalize_markdown_result(page_md, page_number=page_num+1)
                        if page_text and page_text.strip():
                            markdown_parts.append(page_text)
                    except Exception as page_error:
                        print(f"⚠️ ページ {page_num + 1} スキップ: {page_error}")
                        continue
                
                doc.close()
                
                if markdown_parts:
                    result = "\n\n---\n\n".join(markdown_parts)
                    print(f"✅ ページ別変換完了: {len(markdown_parts)}ページ処理")
                    return result
                else:
                    raise ValueError("全ページの変換に失敗しました")
                    
            except Exception as second_error:
                print(f"❌ ページ別変換も失敗: {second_error}")
                raise ValueError(f"PDF変換失敗: {first_error}")

    def _filter_supported_kwargs(self, kwargs: dict) -> dict:
        """to_markdownのシグネチャから受理される引数だけを抽出"""
        try:
            import inspect
            sig = inspect.signature(pymupdf4llm.to_markdown)
            accepted = set(sig.parameters.keys())
            provided = set((kwargs or {}).keys())
            rejected = sorted(provided - accepted)
            if rejected:
                print("⚠️ 未対応のpymupdf4llm引数を除外:", ", ".join(rejected))
            filtered = {k: v for k, v in (kwargs or {}).items() if k in accepted}
            # marginsの型を強制（float or 長さ1/2/4のtuple[float]）
            if "margins" in filtered:
                ok, coerced = self._coerce_margins(filtered.get("margins"))
                if ok:
                    filtered["margins"] = coerced
                else:
                    print("⚠️ margins を無効化（形式不正）:", filtered.get("margins"))
                    filtered.pop("margins", None)
            # 画像出力先を固定フォルダに設定（pdf_converter.py がある new_pdf_converter/images）
            try:
                if filtered.get("write_images") and not filtered.get("embed_images"):
                    base_dir = Path(__file__).resolve().parents[1]  # new_pdf_converter
                    img_dir = base_dir / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    filtered["image_path"] = str(img_dir)
            except Exception as _e:
                print(f"⚠️ 画像出力先の自動設定に失敗: {_e}")
            return filtered
        except Exception:
            # 失敗時はそのまま返す（下流でTypeErrorが出た場合は上位で処理）
            return dict(kwargs or {})


    # --- VLM captioning (streaming with progress) ---
    def _build_vlm_env(self):
        cfg = {
            'nodes': [{'id': '1', 'type': 'vlm'}],
            'node_thresholds': {'1': {'preset': 'balanced'}},
            'logging': {'default_level': 'MINIMAL', 'node_specific_levels': {'1': 'MINIMAL'}},
        }
        log = LogManager(cfg)
        thr = NodeThresholdManager(cfg)
        mm = ModelManager()
        mm.setup_model()
        vlm = VLMExecutor(log, thr, model_manager=mm)
        vlm.set_node_config({}, '1')
        return log, thr, vlm, mm

    def _set_thresholds(self, thr: NodeThresholdManager, params: Dict):
        node_id = '1'
        for k, v in params.items():
            try:
                thr.set_value(node_id, k, v)
            except Exception:
                pass

    def _parse_classification(self, text: str) -> Dict[str, str]:
        import json as _json
        try:
            obj = _json.loads(text)
            if isinstance(obj, dict):
                t = obj.get('type')
                conf = obj.get('confidence', 0.0)
                reason = obj.get('reason', '')
                return {'type': t, 'confidence': str(conf), 'reason': reason}
        except Exception:
            pass
        low = (text or '').lower()
        t = None
        for cand in list(DEFAULT_TYPE_MAP.keys()):
            if cand in low:
                t = cand
                break
        return {'type': t or 'natural_image', 'confidence': '0.3', 'reason': (text or '')[:200]}

    def _emit_progress(self, ev: Dict):
        try:
            if callable(self.progress_cb):
                self.progress_cb(ev)
        except Exception:
            pass

    def _make_xlsx(self, out_dir: Path, results: List[Dict]):
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / 'captions.xlsx'
        try:
            from openpyxl import Workbook
            from openpyxl.drawing.image import Image as XLImage
            wb = Workbook()
            ws = wb.active
            ws.title = 'captions'
            ws.append(['file', 'path', 'type', 'preset', 'caption', 'image'])
            # 画像列の幅と最大貼り付けサイズ（控えめに調整）
            max_w_px, max_h_px = 240, 140
            try:
                ws.column_dimensions['F'].width = max_w_px / 7.0
            except Exception:
                pass
            row = 2
            for r in results:
                ws.append([r.get('file'), r.get('path'), r.get('type'), r.get('preset'), r.get('caption'), ''])
                img_path = r.get('path')
                try:
                    if img_path and Path(img_path).exists():
                        xlimg = XLImage(img_path)
                        try:
                            w0 = getattr(xlimg, 'width', None)
                            h0 = getattr(xlimg, 'height', None)
                            if w0 and h0 and w0 > 0 and h0 > 0:
                                scale = min(max_w_px / float(w0), max_h_px / float(h0), 1.0)
                                xlimg.width = int(w0 * scale)
                                xlimg.height = int(h0 * scale)
                                ws.row_dimensions[row].height = xlimg.height * 0.75
                        except Exception:
                            pass
                        cell = f'F{row}'
                        ws.add_image(xlimg, cell)
                except Exception:
                    pass
                row += 1
            wb.save(str(xlsx_path))
            return True, str(xlsx_path)
        except Exception as e:
            print(f'⚠️ XLSX出力に失敗しました: {e}')
            # フォールバック: CSV
            try:
                import csv
                csv_path = out_dir / 'captions.csv'
                with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=['file', 'path', 'type', 'preset', 'caption'])
                    w.writeheader()
                    for r in results:
                        w.writerow({k: r.get(k, '') for k in ['file', 'path', 'type', 'preset', 'caption']})
                return False, str(csv_path)
            except Exception as e2:
                print(f'❌ CSV出力にも失敗しました: {e2}')
                return False, None

    def _clear_images_dir(self, images_dir: Path):
        try:
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            for p in images_dir.glob('*'):
                if p.suffix.lower() in exts:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _classify_and_caption_stream(self, images_dir: Path, out_dir: Path, markdown_text: str = "") -> Dict[str, str]:
        """Classify and caption images with context-aware prompts.
        
        Args:
            images_dir: Directory containing extracted images
            out_dir: Output directory for caption results
            markdown_text: Full markdown text to extract context from (optional)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        _, thr, vlm, _ = self._build_vlm_env()
        # 画像収集
        imgs: List[Path] = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp'):
            imgs.extend(sorted(images_dir.glob(ext)))

        results: List[Dict] = []
        path_to_caption: Dict[str, str] = {}

        total = len(imgs)
        for idx, p in enumerate(imgs, start=1):
            # 1) classify
            self._emit_progress({'stage': 'classify_start', 'index': idx, 'total': total, 'file': p.name, 'path': str(p)})
            self._set_thresholds(thr, CLASSIFIER_PARAMS.to_clean_dict())
            resp = vlm.execute({KeyRegistry.USER_QUERY: CLASSIFIER_PROMPT, KeyRegistry.IMAGE_PATH: str(p)}, node_id='1')
            if resp.has_error:
                ctype = 'natural_image'
                parsed = {'type': ctype, 'confidence': '0.0', 'reason': resp.data.get('error', '')}
            else:
                text = resp.data.get(KeyRegistry.VLM_ANSWER, '')
                parsed = self._parse_classification(text)
                ctype = parsed.get('type') or 'natural_image'
                if ctype not in DEFAULT_TYPE_MAP:
                    ctype = 'natural_image'
            self._emit_progress({'stage': 'classify_done', 'file': p.name, 'path': str(p), 'type': ctype, 'info': parsed})

            # 2) extract context around this image (if enabled)
            context = ""
            use_context = getattr(self.settings, 'use_context_for_captions', True)
            if use_context and markdown_text:
                context = self._extract_image_context(markdown_text, p, context_chars=500)
                if context:
                    print(f"✅ 文脈抽出成功: {p.name} - {len(context)}文字")
                    # Show first 200 chars of context for debugging
                    preview = context[:200].replace('\n', ' ')
                    print(f"   📄 文脈プレビュー: {preview}...")
                else:
                    print(f"⚠️ 文脈抽出失敗: {p.name} - 画像参照がmarkdown内に見つかりません")
            else:
                if not use_context:
                    print(f"ℹ️ 文脈判断OFF: {p.name}")
                else:
                    print(f"⚠️ markdown_textなし: {p.name}")

            # 3) caption with context
            entry = DEFAULT_TYPE_MAP[ctype]
            caption_params = entry['params'].to_clean_dict()
            preset = caption_params.get('preset')
            # Emit progress with context indicator
            self._emit_progress({
                'stage': 'caption_start', 
                'file': p.name, 
                'path': str(p), 
                'type': ctype, 
                'preset': preset,
                'context_used': bool(context),
                'context_length': len(context) if context else 0
            })
            self._set_thresholds(thr, caption_params)
            
            # Build context-aware prompt (if context is available)
            base_prompt = entry['prompt']
            if context:
                # NEW FORMAT: Instructions first, context as supporting reference at end
                # This prevents context from overwhelming the main task instruction
                enhanced_prompt = f"""{base_prompt}

ADDITIONAL CONTEXT from the document:
The image appears in the following context:
{context}

Please use this context to provide a more accurate and relevant description."""
                print(f"🔍 文脈付きプロンプト使用: {len(enhanced_prompt)}文字")
                # Show first 300 chars of actual prompt being sent
                print(f"   🎯 送信プロンプト冒頭: {enhanced_prompt[:300]}...")
            else:
                enhanced_prompt = base_prompt
                print(f"📝 通常プロンプト使用: {len(enhanced_prompt)}文字")
            
            cap_resp = vlm.execute({KeyRegistry.USER_QUERY: enhanced_prompt, KeyRegistry.IMAGE_PATH: str(p)}, node_id='1')
            caption = cap_resp.data.get(KeyRegistry.VLM_ANSWER, '') if not cap_resp.has_error else ''
            self._emit_progress({'stage': 'caption_done', 'file': p.name, 'path': str(p), 'type': ctype, 'preset': preset, 'caption': caption[:200]})

            results.append({
                'file': p.name,
                'path': str(p.resolve()),
                'type': ctype,
                'preset': preset,
                'caption': caption,
                'context': context[:100] if context else '',  # Store first 100 chars for reference
            })
            path_to_caption[str(p.resolve())] = caption

        # 出力: XLSX（失敗時CSV）
        ok, out_path = self._make_xlsx(out_dir, results)
        self._emit_progress({'stage': 'export', 'format': 'xlsx' if ok else 'csv', 'path': out_path})

        # 画像フォルダをクリア
        self._clear_images_dir(images_dir)
        self._emit_progress({'stage': 'cleanup', 'path': str(images_dir)})

        return path_to_caption

    def _extract_image_context(self, markdown_text: str, image_path: Path, context_chars: int = 500) -> str:
        """Extract surrounding text context for an image from markdown.
        
        Args:
            markdown_text: Full markdown text
            image_path: Path to the image file
            context_chars: Number of characters to extract before and after
        
        Returns:
            Extracted context text (before + after the image reference)
        """
        if not markdown_text or not image_path:
            return ""
        
        import re
        
        def norm(s: str) -> str:
            """Normalize path for matching."""
            return (s or "").replace("\\", "/").lower().strip()
        
        # Build patterns to find this image in markdown
        img_name = image_path.name
        norm_img_name = norm(img_name)
        
        # Try to find image reference in markdown
        # Patterns: ![...](path), <img src="path">, etc.
        md_pattern = re.compile(r"!\[[^\]]*\]\([^)]*" + re.escape(img_name) + r"[^)]*\)", re.IGNORECASE)
        html_pattern = re.compile(r"<img[^>]*src=['\"][^'\"]*" + re.escape(img_name) + r"[^'\"]*['\"][^>]*>", re.IGNORECASE)
        
        match = md_pattern.search(markdown_text)
        if not match:
            match = html_pattern.search(markdown_text)
        
        if not match:
            # Fallback: try to find just the filename
            simple_pattern = re.compile(re.escape(img_name), re.IGNORECASE)
            match = simple_pattern.search(markdown_text)
        
        if not match:
            return ""  # Image reference not found in markdown
        
        # Extract context around the match
        match_pos = match.start()
        
        # Get text before and after
        start_pos = max(0, match_pos - context_chars)
        end_pos = min(len(markdown_text), match.end() + context_chars)
        
        before_text = markdown_text[start_pos:match_pos].strip()
        after_text = markdown_text[match.end():end_pos].strip()
        
        # Clean up: remove excessive newlines, markdown formatting clutter
        before_text = re.sub(r'\n{3,}', '\n\n', before_text)
        after_text = re.sub(r'\n{3,}', '\n\n', after_text)
        
        # Build context string
        context_parts = []
        if before_text:
            context_parts.append(f"[Text before image]: {before_text[-400:] if len(before_text) > 400 else before_text}")
        if after_text:
            context_parts.append(f"[Text after image]: {after_text[:400] if len(after_text) > 400 else after_text}")
        
        return "\n\n".join(context_parts) if context_parts else ""



    def _coerce_margins(self, val):
        """marginsを to_markdown が期待する形式へ矯正"""
        try:
            # 単一数値
            if isinstance(val, (int, float)):
                return True, float(val)
            # 文字列（例: "20,15"）
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(',') if p.strip()]
                if len(parts) == 1:
                    return True, float(parts[0])
                if len(parts) in (2, 4):
                    return True, tuple(float(x) for x in parts)
                return False, None
            # 配列/タプル
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return True, float(val[0])
                if len(val) in (2, 4):
                    return True, tuple(float(x) for x in val)
                return False, None
            return False, None
        except Exception:
            return False, None

    def _normalize_markdown_result(self, result, page_number: int = None) -> str:
        """pymupdf4llmの戻り値（文字列 or ページ辞書リスト）をMarkdown文字列に正規化"""
        try:
            if isinstance(result, str):
                return result
            # 期待構造: list[dict] で各要素に 'markdown' or 'text'、'page_number' など
            parts = []
            for idx, item in enumerate(result or []):
                md = (item.get('markdown') if isinstance(item, dict) else None) or \
                     (item.get('text') if isinstance(item, dict) else None) or ""
                # ページ番号が提供されない場合はリストの並び順から補完（1-based）
                pg = None
                if isinstance(item, dict):
                    pg = item.get('page_number')
                    if pg is None:
                        # 他の候補キー（ライブラリ差異の保険）
                        pg = item.get('page') or item.get('number') or item.get('page_no')
                if pg is None:
                    pg = page_number if page_number is not None else (idx + 1)
                if pg is not None:
                    parts.append(f"# Page {int(pg)}\n\n{md}")
                else:
                    parts.append(md)
            return "\n\n".join(parts)
        except Exception:
            # 予期しない構造はそのまま文字列化
            return str(result)

    def _debug_inspect_page_chunks(self, pages, label: str = "to_markdown"):
        """ページ配列の要素構造を簡易ダンプして、page_number等の存在を確認する"""
        try:
            if not isinstance(pages, list) or not pages:
                print(f"🧪 [{label}] inspect: pages is empty or not a list")
                return
            sample_idx = [0, min(1, len(pages)-1), len(pages)-1]
            seen_keys = set()
            found_page_id = False
            for i in sample_idx:
                item = pages[i]
                if isinstance(item, dict):
                    keys = list(item.keys())
                    seen_keys.update(keys)
                    pg = item.get('page_number') or item.get('page') or item.get('number') or item.get('page_no')
                    if pg is not None:
                        found_page_id = True
                    md = item.get('markdown') or item.get('text')
                    md_len = len(md) if isinstance(md, str) else 0
                    has_words = 'words' in item and isinstance(item.get('words'), (list, tuple))
                    has_images = 'images' in item and isinstance(item.get('images'), (list, tuple))
                    has_tables = 'tables' in item and isinstance(item.get('tables'), (list, tuple))
                    print(f"🔎 [{label}] sample idx={i+1}: keys={keys[:8]}... md_len={md_len} page_id={pg} words={has_words} images={has_images} tables={has_tables}")
                else:
                    print(f"🔎 [{label}] sample idx={i+1}: non-dict item type={type(item)}")
            print(f"🔎 [{label}] aggregate keys(sampled)={sorted(list(seen_keys))[:12]} ...; page_id_found={found_page_id}")
        except Exception as e:
            print(f"⚠️ debug inspect failed: {e}")
    
    def _create_error_fallback(self, pdf_path: str, error_message: str) -> ProcessedDocument:
        """エラー時のフォールバック処理結果作成"""
        try:
            doc_metadata = self._create_metadata(pdf_path)
            doc_metadata.processor_version += " (ERROR_FALLBACK)"
            
            error_content = f"PDF処理エラーが発生しました。\n\nエラー: {error_message}\n\nファイル: {pdf_path}"
            
            error_chunk = self.chunker._create_fallback_chunk(error_content, doc_metadata)
            
            return ProcessedDocument(
                document_metadata=doc_metadata,
                chunks=error_chunk,
                raw_markdown=error_content,
                processing_stats={"error": True, "error_message": error_message}
            )
            
        except Exception as e:
            print(f"❌ フォールバック作成も失敗: {e}")
            raise
    
    def _calculate_stats_safe(self, markdown_text: str, chunks) -> Dict:
        """統計計算（安全版・ゼロ除算対策）"""
        try:
            total_chunks = len(chunks) if chunks else 0
            total_chars = len(markdown_text) if markdown_text else 0
            
            # ゼロ除算対策
            if total_chunks > 0:
                avg_chunk_size = sum(c.chunk_metadata.char_count for c in chunks if c and hasattr(c, 'chunk_metadata')) / total_chunks
                chunk_types = self._count_chunk_types(chunks)
                formula_chunks = sum(1 for c in chunks if c and hasattr(c, 'chunk_metadata') and c.chunk_metadata.contains_formulas)
                table_chunks = sum(1 for c in chunks if c and hasattr(c, 'chunk_metadata') and c.chunk_metadata.contains_tables)
                code_chunks = sum(1 for c in chunks if c and hasattr(c, 'chunk_metadata') and c.chunk_metadata.contains_code)
            else:
                avg_chunk_size = 0
                chunk_types = {}
                formula_chunks = 0
                table_chunks = 0
                code_chunks = 0
            
            return {
                "total_chunks": total_chunks,
                "total_chars": total_chars,
                "avg_chunk_size": avg_chunk_size,
                "chunk_types": chunk_types,
                "formula_chunks": formula_chunks,
                "table_chunks": table_chunks,
                "code_chunks": code_chunks
            }
            
        except Exception as e:
            print(f"⚠️ 統計計算エラー: {e}")
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "chunk_types": {},
                "formula_chunks": 0,
                "table_chunks": 0,
                "code_chunks": 0,
                "error": str(e)
            }
    
    # 残りのメソッドは既存のまま...
    def _create_metadata(self, pdf_path: str) -> DocumentMetadata:
        """メタデータ作成"""
        file_path = Path(pdf_path)
        
        # ファイルハッシュ
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        return DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path.absolute()),
            file_size=file_path.stat().st_size,
            source_hash=file_hash,
            document_type=self._detect_document_type(file_path.name)
        )
    
    def _detect_document_type(self, filename: str) -> str:
        """文書タイプ検出"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['manual', 'guide', 'handbook']):
            return "manual"
        elif any(word in filename_lower for word in ['paper', 'journal', 'conference']):
            return "technical_paper"
        elif any(word in filename_lower for word in ['api', 'reference', 'doc']):
            return "reference"
        elif 'modelica' in filename_lower:
            return "modelica_document"
        else:
            return "general"
    
    def _count_chunk_types(self, chunks) -> Dict[str, int]:
        """チャンクタイプ集計（安全版）"""
        try:
            type_counts = {}
            for chunk in chunks:
                if chunk and hasattr(chunk, 'chunk_metadata'):
                    chunk_type = chunk.chunk_metadata.chunk_type
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            return type_counts
        except Exception:
            return {}

    def _apply_rag_metadata(self, chunks, doc_metadata: DocumentMetadata):
        """RAGメタ付与は廃止（何もしない）"""
        return

    def _replace_markdown_images(self, markdown_text: str, mapping: Dict[str, str]) -> str:
        """本番用: Markdown/HTMLの画像参照を、captions.xlsx と同一キー（正規化済みパス）で
        1:1 のキャプションに置換する。

        - Markdown: ![alt](path), ![alt]("path"), ![alt]('path') に対応
        - HTML: <img src="path"> と <img src='path'> に対応
        - 照合はパスを正規化（区切りを/化・小文字化・前後空白除去）して行う
        """
        import re

        def norm(s: str) -> str:
            return (s or "").replace("\\", "/").lower().strip()

        # 正規化済みのキーでマップを作り直す
        norm_map: Dict[str, str] = {norm(k): v for k, v in mapping.items() if k}

        out = markdown_text

        # Markdown image: ![alt](url) or quoted url
        md_pat = re.compile(r"!\[[^\]]*\]\(\s*(?P<url>\"[^\"]+\"|'[^']+'|[^\s)]+)\s*\)")

        def _md_repl(m: re.Match) -> str:
            raw = m.group("url")
            if raw and ((raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'"))):
                url = raw[1:-1]
            else:
                url = raw
            cap = norm_map.get(norm(url))
            if cap is not None:
                # Wrap caption in markers for LLM parsing
                return f"<!-- IMAGE_CAPTION_START -->\n{cap}\n<!-- IMAGE_CAPTION_END -->"
            else:
                return m.group(0)

        out = md_pat.sub(_md_repl, out)

        # HTML image: <img ... src="url" ...> or src='url'
        html_pat = re.compile(r"<img[^>]*?src=\"(?P<url>[^\"]+)\"[^>]*?>|<img[^>]*?src='(?P<url2>[^']+)'[^>]*?>", re.IGNORECASE)

        def _html_repl(m: re.Match) -> str:
            url = m.group("url") if m.groupdict().get("url") else m.group("url2")
            cap = norm_map.get(norm(url))
            if cap is not None:
                # Wrap caption in markers for LLM parsing
                return f"<!-- IMAGE_CAPTION_START -->\n{cap}\n<!-- IMAGE_CAPTION_END -->"
            else:
                return m.group(0)

        out = html_pat.sub(_html_repl, out)

        return out






