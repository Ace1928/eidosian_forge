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
def configure_resource_slider(self) -> None:
    """Configures the resource usage slider based on current settings, ensuring intuitive user control."""
    self.resource_usage_slider.setMinimum(10)
    self.resource_usage_slider.setMaximum(100)
    resource_usage = self.settings_manager.settings.get('resource_usage', 50)
    self.resource_usage_slider.setValue(resource_usage)
    self.resource_usage_slider.valueChanged[int].connect(self.adjust_resource_usage)