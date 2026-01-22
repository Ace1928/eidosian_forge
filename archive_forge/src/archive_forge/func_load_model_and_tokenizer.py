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
def load_model_and_tokenizer(self, model_id: str=None) -> None:
    """Load model and tokenizer, preferring local cache if available.

        Args:
            model_id (str): Identifier for the model to load.
        """
    log_verbose(f'Initiating model and tokenizer loading for {model_id}')
    try:
        model_path = self.settings_manager.settings.get('model_path', None)
        if model_path and os.path.exists(model_path):
            log_verbose('Loading model from local cache.')
            self.model_pipeline = pipeline('text-generation', model=model_path, tokenizer=model_path, device=0 if device == 'cuda' else -1)
        else:
            log_verbose('Loading model from Hugging Face Hub.')
            self.model_pipeline = pipeline('text-generation', model=model_id, device=0 if device == 'cuda' else -1)
    except Exception as error:
        log_verbose(f'Model loading error: {error}')
        raise