# new_pdf_converter/pymupdf_converter/batch_config_dialog.py
"""Batch processing configuration dialog."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QCheckBox, QLineEdit, QPushButton, QGroupBox, QButtonGroup,
    QListWidget, QFormLayout, QFileDialog
)
from PySide6.QtCore import Qt
from pathlib import Path


class BatchConfigDialog(QDialog):
    """バッチ処理設定ダイアログ"""
    
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.output_folder = str(Path(file_paths[0]).parent)  # Default to first file's folder
        self.setWindowTitle("バッチ処理設定")
        self.setMinimumSize(500, 500)
        self._setup_ui()
        self._update_preview()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # ファイル一覧
        files_group = QGroupBox(f"処理対象ファイル ({len(self.file_paths)}件)")
        files_layout = QVBoxLayout(files_group)
        self.file_list = QListWidget()
        for path in self.file_paths:
            self.file_list.addItem(Path(path).name)
        files_layout.addWidget(self.file_list)
        layout.addWidget(files_group)
        
        # 出力形式
        format_group = QGroupBox("出力形式")
        format_layout = QHBoxLayout(format_group)
        self.format_group = QButtonGroup()
        self.md_radio = QRadioButton("Markdown (.md)")
        self.json_radio = QRadioButton("JSON (.json)")
        self.md_radio.setChecked(True)
        self.format_group.addButton(self.md_radio, 0)
        self.format_group.addButton(self.json_radio, 1)
        format_layout.addWidget(self.md_radio)
        format_layout.addWidget(self.json_radio)
        layout.addWidget(format_group)
        
        # 命名規則
        naming_group = QGroupBox("命名規則")
        naming_layout = QFormLayout(naming_group)
        
        # プレフィックス
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("例: converted_")
        self.prefix_input.textChanged.connect(self._update_preview)
        naming_layout.addRow("プレフィックス:", self.prefix_input)
        
        # サフィックス追加
        self.suffix_check = QCheckBox("サフィックスを追加")
        self.suffix_check.stateChanged.connect(self._on_suffix_toggled)
        naming_layout.addRow(self.suffix_check)
        
        # サフィックスタイプ
        suffix_type_layout = QHBoxLayout()
        self.suffix_type_group = QButtonGroup()
        self.number_radio = QRadioButton("連番 (001, 002...)")
        self.alpha_radio = QRadioButton("アルファベット (a, b, c...)")
        self.number_radio.setChecked(True)
        self.number_radio.toggled.connect(self._update_preview)
        self.alpha_radio.toggled.connect(self._update_preview)
        self.suffix_type_group.addButton(self.number_radio, 0)
        self.suffix_type_group.addButton(self.alpha_radio, 1)
        suffix_type_layout.addWidget(self.number_radio)
        suffix_type_layout.addWidget(self.alpha_radio)
        suffix_type_layout.addStretch()
        naming_layout.addRow("  ", suffix_type_layout)
        
        # 初期状態: サフィックスオプション無効
        self.number_radio.setEnabled(False)
        self.alpha_radio.setEnabled(False)
        
        layout.addWidget(naming_group)
        
        # 保存先フォルダ
        folder_group = QGroupBox("保存先フォルダ")
        folder_layout = QHBoxLayout(folder_group)
        self.folder_label = QLabel(self.output_folder)
        self.folder_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; }")
        self.folder_label.setWordWrap(True)
        self.browse_btn = QPushButton("参照...")
        self.browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(self.browse_btn)
        layout.addWidget(folder_group)
        
        # プレビュー
        preview_group = QGroupBox("出力ファイル名プレビュー")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_list = QListWidget()
        self.preview_list.setMaximumHeight(100)
        preview_layout.addWidget(self.preview_list)
        layout.addWidget(preview_group)
        
        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("キャンセル")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
    def _browse_folder(self):
        """保存先フォルダ選択"""
        folder = QFileDialog.getExistingDirectory(
            self, "保存先フォルダを選択", self.output_folder
        )
        if folder:
            self.output_folder = folder
            self.folder_label.setText(folder)
    
    def _on_suffix_toggled(self, state):
        """サフィックスチェックボックスの状態変更"""
        enabled = bool(state)
        self.number_radio.setEnabled(enabled)
        self.alpha_radio.setEnabled(enabled)
        self._update_preview()
        
    def _update_preview(self):
        """プレビュー更新"""
        self.preview_list.clear()
        
        ext = ".md" if self.md_radio.isChecked() else ".json"
        prefix = self.prefix_input.text()
        add_suffix = self.suffix_check.isChecked()
        use_number = self.number_radio.isChecked()
        
        for idx, path in enumerate(self.file_paths):
            base = Path(path).stem
            
            # プレフィックス
            if prefix:
                name = prefix + base
            else:
                name = base
            
            # サフィックス
            if add_suffix:
                if use_number:
                    name += f"_{idx+1:03d}"
                else:
                    name += f"_{chr(97+idx)}" if idx < 26 else f"_{idx}"
            
            self.preview_list.addItem(name + ext)
    
    def get_config(self):
        """設定を取得"""
        return {
            'format': 'markdown' if self.md_radio.isChecked() else 'json',
            'prefix': self.prefix_input.text(),
            'add_suffix': self.suffix_check.isChecked(),
            'suffix_type': 'number' if self.number_radio.isChecked() else 'alphabet',
            'output_folder': self.output_folder
        }
