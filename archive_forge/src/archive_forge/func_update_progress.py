import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes
def update_progress(self, value: int) -> None:
    """
        Updates the progress bar value.

        Args:
            value (int): The progress value to set.
        """
    try:
        self.progress_bar.setValue(value)
    except Exception as e:
        logger.error(f'Error updating progress bar: {e}')