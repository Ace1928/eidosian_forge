import sys
import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget,
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import json
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from huggingface_hub import HfApi
def populate_model_list(self) -> None:
    """Populates the model list from the Hugging Face model repository with robust error handling."""
    try:
        api: HfApi = HfApi()
        models = api.list_models(filter=('pytorch',), sort='downloads')
        for model in models:
            self.model_list.addItem(f'{model.modelId} - Downloads: {model.downloads}')
    except Exception as e:
        log_verbose(f'Error fetching models: {e}')
        QMessageBox.critical(self, 'Error', 'Failed to fetch model list.')