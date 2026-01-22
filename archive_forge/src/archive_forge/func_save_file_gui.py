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
def save_file_gui(title: str, file_type: str) -> Path:
    """
    Opens an enhanced file dialog to specify the path and name of a file to be saved, with better user feedback.

    Args:
        title (str): The title of the file dialog window.
        file_type (str): The type of file to be saved, e.g., 'WAV files (*.wav)'.

    Returns:
        Path: The path where the file will be saved.
    """
    root = Tk()
    root.withdraw()
    file_path: Path = filedialog.asksaveasfilename(title=title, defaultextension=f'.{file_type.split()[-1]}', filetypes=[(file_type, f'*.{file_type.split()[-1]}')])
    if file_path:
        logging.info(f'File to be saved at: {file_path}')
    else:
        logging.info('File save cancelled.')
    return file_path