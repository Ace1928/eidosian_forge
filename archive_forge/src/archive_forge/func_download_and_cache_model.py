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
def download_and_cache_model(model_id: str, cache_dir: str='./models') -> str:
    """Downloads and caches the model locally.

    Args:
        model_id (str): Identifier for the model to download.
        cache_dir (str): Directory to store the cached models.

    Returns:
        str: Path to the cached model.
    """
    logger.info(f'Downloading and caching model: {model_id}')
    model_path = os.path.join(cache_dir, model_id.replace('/', '_'))
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.save_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
    return model_path