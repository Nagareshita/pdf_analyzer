# PDF Parser Tab - Standalone Package

このパッケージは `pdf_analyzer.py` を別の環境で動作させるために必要なファイルをすべて含んでいます。

## 📁 フォルダ構成

```
new/
├── pdf_analyzer.py          # メインタブファイル
├── requirements.txt           # 必要なPythonパッケージ
├── README.md                  # このファイル
├── new_pdf_converter/         # PDFコンバーターモジュール
│   ├── pymupdf_converter/    # PDF処理の中核
│   ├── pdf_converter_setting.json  # 設定スキーマ
│   └── requirements.txt       # モジュール固有の依存関係
├── utils/                     # ユーティリティモジュール
│   ├── agents/               # エージェント実行クラス
│   ├── log_manager.py        # ログマネージャー
│   ├── key_registry.py       # キーレジストリ
│   └── その他のユーティリティ
└── vlm/                       # Vision Language Model
    ├── model_manager.py      # モデル管理
    ├── caption_maker/        # 画像キャプション生成
    └── SAIL-VL2-2B/         # VLMモデル（要ダウンロード）
```

## 🚀 セットアップ手順（推奨: Mamba + Python 3.12）

### 前提条件

- **Mamba** または **Conda** がインストールされていること
  - Mambaは高速なパッケージ管理ツールです（Condaの上位互換）
  - まだインストールしていない場合: [Miniforge](https://github.com/conda-forge/miniforge) からインストール
- **NVIDIA GPU**（推奨）
  - CPU でも動作しますが、VLM（画像キャプション）は非常に遅くなります

---

### ① 仮想環境の作成

Mambaを使って、Python 3.12の仮想環境を作成します。

```powershell
# 環境名: vlm, Python 3.12
mamba create -n vlm python=3.12 -y
```

### ② 仮想環境の有効化

作成した環境を有効化します。

```powershell
conda activate vlm
```

**確認:** プロンプトが `(vlm)` に変わっていればOKです。

---

### ③ PyTorch（CUDA版）のインストール

**重要:** 先にPyTorchをmambaでインストールすることで、CUDA依存関係が正しく解決されます。

```powershell
mamba install "pytorch=*=*cuda*"  -y
```

**補足:**
- `pytorch-cuda=12.4` の部分は、お使いのGPUのCUDAバージョンに合わせてください
  - CUDA 11.8の場合: `pytorch-cuda=11.8`
  - CUDA 12.1の場合: `pytorch-cuda=12.1`
- CUDA バージョンは `nvidia-smi` コマンドで確認できます

**CPU版の場合（GPU不要・動作遅い）:**

```powershell
mamba install pytorch torchvision cpuonly
```

---

### ④ uvのインストール

パッケージ管理ツール `uv` をインストールします（高速なpip代替ツール）。

```powershell
pip install uv
```

**確認:**

```powershell
uv --version
```

---

### ⑤ その他の依存パッケージのインストール

プロジェクトのルートディレクトリ（`new`フォルダ）に移動し、`requirements.txt`からインストールします。

```powershell
uv pip install -r requirements.txt
```

**注意:** PyTorchは既にmambaでインストールされているため、この手順ではスキップされます。

---

### ⑥ 動作確認

VLM（画像キャプション機能）が正しくセットアップされたか確認します。

```powershell
python verify_vlm.py
```

**成功メッセージ:**

```
✅ Model setup successful!
```

が表示されればOKです。

---

### ⑦ アプリケーションの起動

PDF Parser を起動します。

```powershell
python pdf_analyzer.py
```

GUIウィンドウが開けば、セットアップ完了です！

---

## 💡 トラブルシューティング

### エラー: `Torch not compiled with CUDA enabled`

原因: CPU版のPyTorchがインストールされています。

**解決策:**

1. PyTorchを一度アンインストール:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```

2. ③の手順でCUDA版を再インストール:
   ```powershell
   mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

### エラー: `pyside6 was not found`

原因: `--index-url` の設定ミスでPyPIが参照できていません。

**解決策:**

`requirements.txt` の1行目が `--extra-index-url` になっているか確認してください（`--index-url` ではない）。

### VLMモデルのダウンロードが進まない

初回起動時、SAIL-VL2-2Bモデル（約4GB）が自動ダウンロードされます。ネットワーク環境によっては時間がかかります。

**手動ダウンロード:**

```powershell
huggingface-cli download BytedanceDouyinContent/SAIL-VL2-2B --local-dir vlm/SAIL-VL2-2B
```



## 📝 使用方法

### スタンドアロンアプリとして実行

```python
from PySide6.QtWidgets import QApplication
import sys

# プロジェクトルートをパスに追加（必要に応じて）
# sys.path.insert(0, "/path/to/new")

from pdf_analyzer import PDFParserTab

app = QApplication(sys.argv)
window = PDFParserTab()
window.show()
sys.exit(app.exec())
```

### 既存のPySide6アプリに組み込み

```python
from pdf_analyzer import PDFParserTab

# タブウィジェットに追加
tab_widget.addTab(PDFParserTab(), "PDF Parser")
```

## ⚙️ 設定

### PDF変換設定

`new_pdf_converter/pdf_converter_setting.json` でPDF変換の詳細設定を変更できます：

- **page_chunks**: ページごとのチャンク化
- **write_images**: 画像をファイルとして保存
- **generate_captions**: 画像キャプションを自動生成（VLM使用）
- **table_strategy**: 表の検出方式
- その他多数のパラメータ

### VLMパラメータ

画像キャプション生成には以下のプリセットが用意されています：

- `accurate`: 正確性重視（OCR向け）
- `balanced`: バランス型（画像説明向け）
- `ocr`: テキスト抽出特化
- `qa`: 質問応答形式
- `code`: コード/図表分析
- `creative`: 創造的な説明
- `summary`: 要約特化
- `json`: JSON形式出力

## 🔧 トラブルシューティング

### CUDA/GPU関連エラー

VLMモデルは大量のメモリを必要とします。GPUメモリが不足する場合：

1. より小さい `max_new_tokens` を設定
2. `num_beams` を減らす（1にするとGreedy Searchに）
3. CPUモードで実行（遅いですが動作します）

### モジュールインポートエラー

```python
# pdf_analyzer.py の冒頭を確認
sys.path.insert(0, str(project_root))
```

パスが正しく設定されているか確認してください。

### PDFからテキストが抽出できない

- PDF が画像ベースの場合、OCRが必要です（pymupdf4llm は標準ではOCR未対応）
- `force_text=True` を設定してみてください
- テーブルが崩れる場合は `table_strategy` を変更してください

## 📦 依存関係の詳細

主要な依存パッケージ：

- **PySide6**: GUIフレームワーク
- **pymupdf4llm**: PDF→Markdown変換
- **torch + torchvision + transformers**: VLMモデル実行
- **Pillow**: 画像処理
- **openpyxl**: XLSX出力（キャプション結果）
- **qdrant-client**: ベクトルデータベース（utils依存）
- **langgraph**: ワークフローエンジン（utils依存）

## 🌟 主な機能

1. **PDF → Markdown 変換**: pymupdf4llm による高品質変換
2. **チャンク分割**: RAG向けの最適なチャンクサイズで分割
3. **画像キャプション生成**: VLMによる自動画像説明（オプション）
4. **統計表示**: チャンク数、タイプ分布、文字数など
5. **エクスポート**: JSON/Markdown形式での保存

## 📄 ライセンス

このパッケージは元のRAG_systemプロジェクトから抽出されたものです。
各モジュールのライセンスに従ってください。

## 🤝 サポート

問題が発生した場合は、元のプロジェクトのIssueトラッカーを参照してください。

---

**注意**: SAIL-VL2-2Bモデルは約4GBのディスク容量と、実行時に8GB以上のGPUメモリを必要とします。
画像キャプション機能を使わない場合は、VLMモデルのダウンロードは不要です。
