import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import wave
import contextlib
import srt
import datetime
import asyncio
from typing import Optional, Tuple, List, Callable, Awaitable, Union, Any, Dict
import os
import aiofiles
import requests
from tkinter import filedialog, Tk, simpledialog
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import logging
import torch
import torchaudio
import soundfile as sf
import googletrans
from googletrans import Translator
def initUI(self) -> None:
    self.setWindowTitle('Download Audio Tracks')
    self.setGeometry(100, 100, 600, 400)
    layout = QVBoxLayout()
    self.browser = QWebEngineView()
    self.browser.setUrl(QUrl('https://www.bensound.com/'))
    self.downloadButton = QPushButton('Download Selected Track')
    self.downloadButton.clicked.connect(self.downloadTrack)
    layout.addWidget(self.browser)
    layout.addWidget(self.downloadButton)
    self.setLayout(layout)