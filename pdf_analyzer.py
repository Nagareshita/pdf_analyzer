# tabs/pdf_analyzer.py
"""
PDF解析タブ
pdf_converterフォルダのロジックを移植
"""

import sys
from pathlib import Path
import json
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QWidget,
)
from PySide6.QtCore import Qt

# 依存関係をパスに追加（単体実行時のパス解決用）
current_dir = Path(__file__).resolve().parent
repo_root = current_dir
sys.path.insert(0, str(repo_root))

from new_pdf_converter.pymupdf_converter.llm_models import ProcessedDocument
from new_pdf_converter.pymupdf_converter.control_panel import ControlPanel
from new_pdf_converter.pymupdf_converter.result_viewer import ResultViewer

class PDFParserTab(QWidget):
    """PDF解析タブ"""
    
    def __init__(self):
        super().__init__()
        self.current_result = None
        self.worker = None
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """UI構築"""
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # コントロールパネル（横幅を狭く）
        self.control_panel = ControlPanel()
        # self.control_panel.setMaximumWidth(450)  # 最大幅制限を解除
        splitter.addWidget(self.control_panel)

        # 結果ビューアー
        self.result_viewer = ResultViewer()
        splitter.addWidget(self.result_viewer)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3) 

        layout.addWidget(splitter)
    
    def _connect_signals(self):
        """シグナル接続"""
        self.control_panel.processing_requested.connect(self._start_processing)
        self.control_panel.batch_processing_requested.connect(self._start_batch_processing)
        self.result_viewer.save_requested.connect(self._save_results)
    
    def _start_processing(self, settings: dict):
        """処理開始"""
        self.control_panel.set_processing_state(True)
        
        # VLMプログレスをクリア
        try:
            self.result_viewer.clear_vlm_progress()
        except Exception:
            pass
        
        # ワーカー開始
        # 重い依存関係（pymupdf4llm / torch / transformers など）を
        # アプリ起動時ではなく実行時に読み込むため、ここで遅延インポートする
        from new_pdf_converter.pymupdf_converter.pdf_processor import PDFProcessorWorker
        self.worker = PDFProcessorWorker(settings)
        self.worker.progress_updated.connect(self.control_panel.update_status)
        self.worker.processing_completed.connect(self._on_processing_completed)
        self.worker.error_occurred.connect(self._on_error_occurred)
        # VLM進捗を結果ビューへ反映
        try:
            self.worker.vlm_progress.connect(self._on_vlm_progress)
        except Exception:
            pass
        self.worker.start()

    def _start_batch_processing(self, files, config):
        """バッチ処理開始"""
        self.control_panel.set_processing_state(True)
        
        # VLMプログレスをクリア
        try:
            self.result_viewer.clear_vlm_progress()
        except Exception:
            pass
        
        # Process files sequentially
        from new_pdf_converter.pymupdf_converter.pdf_processor import PDFProcessorWorker
        
        total = len(files)
        self.control_panel.update_status(f"バッチ処理開始: {total}件のファイル")
        
        # Process first file
        self._batch_files = files
        self._batch_config = config
        self._batch_index = 0
        self._batch_results = []
        
        self._process_next_batch_file()
    
    def _process_next_batch_file(self):
        """次のバッチファイルを処理"""
        if self._batch_index >= len(self._batch_files):
            # All files processed
            self._on_batch_completed()
            return
        
        file_path = self._batch_files[self._batch_index]
        self.control_panel.update_status(
            f"処理中 ({self._batch_index+1}/{len(self._batch_files)}): {Path(file_path).name}"
        )
        
        # Build settings for this file
        settings = {
            'pdf_path': file_path,
            'chunk_size': self.control_panel.chunk_size_spin.value(),
            'overlap_size': self.control_panel.overlap_size_spin.value(),
            'pymupdf_kwargs': self.control_panel._collect_pymupdf_kwargs(),
            'generate_captions': self.control_panel.caption_checkbox.isChecked(),
            'use_context_for_captions': self.control_panel.context_caption_checkbox.isChecked(),
        }
        
        from new_pdf_converter.pymupdf_converter.pdf_processor import PDFProcessorWorker
        self.worker = PDFProcessorWorker(settings)
        self.worker.progress_updated.connect(self.control_panel.update_status)
        self.worker.processing_completed.connect(self._on_batch_file_completed)
        self.worker.error_occurred.connect(self._on_batch_file_error)
        # VLM進捗を結果ビューへ反映
        try:
            self.worker.vlm_progress.connect(self._on_vlm_progress)
        except Exception:
            pass
        self.worker.start()
    
    def _on_batch_file_completed(self, result: ProcessedDocument):
        """バッチファイル処理完了"""
        # Save result
        file_path = self._batch_files[self._batch_index]
        output_name = self._generate_output_name(file_path, self._batch_index)
        
        try:
            self._save_batch_result(result, output_name)
            self._batch_results.append((Path(file_path).name, "成功"))
        except Exception as e:
            self._batch_results.append((Path(file_path).name, f"保存失敗: {e}"))
        
        # Process next file
        self._batch_index += 1
        self._process_next_batch_file()
    
    def _on_batch_file_error(self, error_message: str):
        """バッチファイルエラー"""
        file_path = self._batch_files[self._batch_index]
        self._batch_results.append((Path(file_path).name, f"エラー: {error_message}"))
        
        # Continue with next file
        self._batch_index += 1
        self._process_next_batch_file()
    
    def _on_batch_completed(self):
        """バッチ処理完了"""
        self.control_panel.set_processing_state(False)
        self.control_panel.update_status("✅ バッチ処理完了")
        
        # Show results summary
        success_count = sum(1 for _, status in self._batch_results if status == "成功")
        total_count = len(self._batch_results)
        
        summary = f"バッチ処理完了: {success_count}/{total_count}件成功\n\n"
        for filename, status in self._batch_results:
            summary += f"{filename}: {status}\n"
        
        QMessageBox.information(self, "バッチ処理完了", summary)
    
    def _generate_output_name(self, file_path, index):
        """出力ファイル名生成"""
        base = Path(file_path).stem
        config = self._batch_config
        
        if config.get('prefix'):
            base = config['prefix'] + base
        
        if config.get('add_suffix'):
            if config['suffix_type'] == 'number':
                base += f"_{index+1:03d}"
            else:  # alphabet
                base += f"_{chr(97+index)}" if index < 26 else f"_{index}"
        
        ext = ".md" if config['format'] == 'markdown' else ".json"
        return base + ext
    
    def _save_batch_result(self, result: ProcessedDocument, filename: str):
        """バッチ結果保存"""
        output_dir = Path(self._batch_config['output_folder'])
        file_path = output_dir / filename
        
        if self._batch_config['format'] == 'markdown':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result.raw_markdown)
        else:  # json
            data = {
                "document_metadata": result.document_metadata.__dict__,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chunk_metadata": chunk.chunk_metadata.__dict__
                    }
                    for chunk in result.chunks
                ],
                "processing_stats": result.processing_stats
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _on_vlm_progress(self, ev: dict):
        try:
            # キャプション開始のタイミングで自動的にVLMプログレスタブへ切り替え
            if ev.get('stage') == 'caption_start':
                if hasattr(self.result_viewer, 'focus_vlm_tab'):
                    self.result_viewer.focus_vlm_tab()
        except Exception:
            pass
        # 逐次行追加
        try:
            self.result_viewer.append_vlm_event(ev)
        except Exception:
            pass
    
    def _on_processing_completed(self, result: ProcessedDocument):
        """処理完了"""
        self.current_result = result
        
        self.control_panel.set_processing_state(False)
        self.control_panel.update_status("✅ 処理完了")
        
        self.result_viewer.display_results(result)
        
        QMessageBox.information(
            self, "処理完了",
            f"変換が完了しました！\n\n"
            f"総チャンク数: {result.processing_stats['total_chunks']}\n"
            f"平均チャンクサイズ: {result.processing_stats['avg_chunk_size']:.0f}文字"
        )
    
    def _on_error_occurred(self, error_message: str):
        """エラー発生"""
        self.control_panel.set_processing_state(False)
        self.control_panel.update_status("❌ エラー発生")
        
        QMessageBox.critical(self, "処理エラー", error_message)
    
    def _save_results(self, save_type: str, file_path: str):
        """結果保存"""
        if not self.current_result:
            return
        
        try:
            if save_type == "json":
                # ProcessedDocumentをdict変換
                data = {
                    "document_metadata": self.current_result.document_metadata.__dict__,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "chunk_metadata": chunk.chunk_metadata.__dict__
                        }
                        for chunk in self.current_result.chunks
                    ],
                    "processing_stats": self.current_result.processing_stats
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            elif save_type == "markdown":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_result.raw_markdown)
            
            QMessageBox.information(self, "保存完了", f"ファイルを保存しました:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", f"保存に失敗しました:\n{e}")
    
    def closeEvent(self, event):
        """タブ終了時のクリーンアップ"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(3000)
        event.accept()


class PDFParserWindow(QMainWindow):
    """スタンドアロン実行用のラッパーウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Parser")
        self.setMinimumSize(1100, 700)
        self.tab = PDFParserTab()
        self.setCentralWidget(self.tab)

    def closeEvent(self, event):
        # Qtの終了時にワーカーを確実に止める
        try:
            self.tab.closeEvent(event)
        finally:
            super().closeEvent(event)


def main():
    """スタンドアロン起動用エントリポイント"""
    # 高DPI環境でも見やすくする（利用可能な環境のみ）
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    app = QApplication(sys.argv)
    window = PDFParserWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
