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
def async_error_handler(func: AsyncFunction) -> AsyncFunction:
    """
    A decorator to handle exceptions in asynchronous functions, log them, and provide user feedback.

    Args:
        func (AsyncFunction): The asynchronous function to be decorated.

    Returns:
        AsyncFunction: The wrapped asynchronous function with exception handling and logging.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f'An error occurred in {func.__name__}: {e}', exc_info=True)
            raise e
    return wrapper