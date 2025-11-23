# utils/agents/vlm_executor.py
"""
VLM Executor - Vision Language Model エージェント

SAIL-VL2-2Bを使用した画像認識・質問応答エージェント
"""

from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
from .base_agent import BaseAgentExecutor, AgentResult
from utils.log_manager import LogLevel
from utils.key_registry import KeyRegistry


class VLMExecutor(BaseAgentExecutor):
    """Vision Language Model エージェント実行クラス"""
    
    def __init__(self, log_manager, threshold_manager, model_manager=None):
        super().__init__("vlm", log_manager, threshold_manager)
        self.model_manager = model_manager
        self._node_config = {"config": {}, "node_id": "unknown"}
    
    def set_model_manager(self, model_manager):
        """ModelManagerを設定（ワークフロー開始時に1回だけ初期化）"""
        self.model_manager = model_manager
    
    def execute(self, current_data: Dict[str, Any], node_id: Optional[str] = None, **kwargs) -> AgentResult:
        """VLM実行メイン処理"""
        node_id = node_id or self._node_config.get("node_id", "unknown")
        node_id_str = str(node_id)
        
        # VERBOSEレベル判定
        log_name = f"vlm_{node_id_str}"
        is_verbose = self.log.should_log(log_name, LogLevel.VERBOSE)
        
        # ModelManager確認
        if not self.model_manager:
            return self._create_error_result("ModelManagerが初期化されていません", node_id)
        
        # ユーザークエリ取得
        user_query = current_data.get(KeyRegistry.USER_QUERY, "")
        if not user_query:
            return self._create_error_result("ユーザークエリが見つかりません", node_id)
        
        # 画像パス取得（オプション）
        image_path = current_data.get(KeyRegistry.IMAGE_PATH)
        image = None
        
        if image_path:
            # 画像読み込み
            try:
                image_path_obj = Path(image_path)
                if not image_path_obj.exists():
                    self._log(LogLevel.MINIMAL, f"警告: 画像ファイルが見つかりません: {image_path}", node_id)
                else:
                    # 🔥 メモリ最適化: load()で即座にピクセルデータを読み込み
                    image = Image.open(image_path_obj)
                    image.load()  # ファイルハンドルをすぐに閉じる
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if is_verbose:
                        self._log(LogLevel.VERBOSE, f"画像読み込み成功: {image_path} ({image.size[0]}x{image.size[1]})", node_id)
                    else:
                        self._log(LogLevel.MINIMAL, f"画像読み込み: {Path(image_path).name}", node_id)
            
            except Exception as e:
                self._log(LogLevel.MINIMAL, f"画像読み込みエラー: {e}", node_id)
                # 画像なしで続行
        
        # ログ出力
        if is_verbose:
            self._log(LogLevel.VERBOSE, "=== VLM応答生成開始 ===", node_id)
            self._log(LogLevel.VERBOSE, f"プロンプト: {user_query}", node_id)
            self._log(LogLevel.VERBOSE, f"画像: {'あり' if image else 'なし'}", node_id)
        else:
            query_preview = user_query[:30] + "..." if len(user_query) > 30 else user_query
            self._log(LogLevel.MINIMAL, f"VLM生成: {query_preview}", node_id)
        
        # 生成パラメータ取得
        temperature = self._get_threshold("temperature")
        top_p = self._get_threshold("top_p")
        top_k = self._get_threshold("top_k")
        max_new_tokens = self._get_threshold("max_new_tokens")
        min_new_tokens = self._get_threshold("min_new_tokens")
        repetition_penalty = self._get_threshold("repetition_penalty")
        no_repeat_ngram_size = self._get_threshold("no_repeat_ngram_size")
        num_beams = self._get_threshold("num_beams")
        length_penalty = self._get_threshold("length_penalty")
        diversity_penalty = self._get_threshold("diversity_penalty")
        early_stopping = self._get_threshold("early_stopping")
        do_sample = self._get_threshold("do_sample")
        preset = self._get_threshold_safe("preset", "balanced")
        
        if is_verbose:
            self._log(LogLevel.VERBOSE, "=== 生成パラメータ ===", node_id)
            self._log(LogLevel.VERBOSE, f"preset: {preset}", node_id)
            self._log(LogLevel.VERBOSE, f"temperature: {temperature}", node_id)
            self._log(LogLevel.VERBOSE, f"top_p: {top_p}", node_id)
            self._log(LogLevel.VERBOSE, f"top_k: {top_k}", node_id)
            self._log(LogLevel.VERBOSE, f"max_new_tokens: {max_new_tokens}", node_id)
            self._log(LogLevel.VERBOSE, f"min_new_tokens: {min_new_tokens}", node_id)
            self._log(LogLevel.VERBOSE, f"repetition_penalty: {repetition_penalty}", node_id)
            self._log(LogLevel.VERBOSE, f"no_repeat_ngram_size: {no_repeat_ngram_size}", node_id)
            self._log(LogLevel.VERBOSE, f"num_beams: {num_beams}", node_id)
            self._log(LogLevel.VERBOSE, f"length_penalty: {length_penalty}", node_id)
            self._log(LogLevel.VERBOSE, f"diversity_penalty: {diversity_penalty}", node_id)
            self._log(LogLevel.VERBOSE, f"early_stopping: {early_stopping}", node_id)
            self._log(LogLevel.VERBOSE, f"do_sample: {do_sample}", node_id)
        
        # VLM応答生成
        try:
            response = self.model_manager.generate_response(
                text=user_query,
                image=image,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                diversity_penalty=diversity_penalty,
                early_stopping=early_stopping,
                do_sample=do_sample,
                preset=preset
            )
            
            if is_verbose:
                self._log(LogLevel.VERBOSE, f"VLM応答生成完了 ({len(response)}文字)", node_id)
                self._log(LogLevel.VERBOSE, f"応答: {response[:100]}...", node_id)
            else:
                self._log(LogLevel.MINIMAL, f"VLM応答生成完了 ({len(response)}文字)", node_id)
            
            # 結果を格納
            output_data = {
                KeyRegistry.VLM_ANSWER: response
            }
            
            return self._create_result(
                confidence=1.0,
                data=output_data,
                has_error=False,
                status="success"
            )
        
        except Exception as e:
            error_msg = f"VLM応答生成エラー: {str(e)}"
            self._log(LogLevel.MINIMAL, error_msg, node_id)
            if is_verbose:
                import traceback
                self._log(LogLevel.VERBOSE, traceback.format_exc(), node_id)
            
            return self._create_error_result(error_msg, node_id)
        
        finally:
            # 🔥 重要: 画像オブジェクトを確実に解放（メモリリーク防止）
            if image is not None:
                try:
                    image.close()
                    del image
                except:
                    pass
