from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
import aiofiles
import httpx
import numpy as np
from gradio_client import utils as client_utils
from PIL import Image, ImageOps, PngImagePlugin
from gradio import utils, wasm_utils
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.utils import abspath, get_upload_folder, is_in_or_equal
def get_video_length(video_path: str | Path):
    if wasm_utils.IS_WASM:
        raise wasm_utils.WasmUnsupportedError('Video duration is not supported in the Wasm mode.')
    duration = subprocess.check_output(['ffprobe', '-i', str(video_path), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv={}'.format('p=0')])
    duration_str = duration.decode('utf-8').strip()
    duration_float = float(duration_str)
    return duration_float