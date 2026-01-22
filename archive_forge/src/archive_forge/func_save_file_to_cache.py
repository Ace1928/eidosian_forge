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
def save_file_to_cache(file_path: str | Path, cache_dir: str) -> str:
    """Returns a temporary file path for a copy of the given file path if it does
    not already exist. Otherwise returns the path to the existing temp file."""
    temp_dir = hash_file(file_path)
    temp_dir = Path(cache_dir) / temp_dir
    temp_dir.mkdir(exist_ok=True, parents=True)
    name = client_utils.strip_invalid_filename_characters(Path(file_path).name)
    full_temp_file_path = str(abspath(temp_dir / name))
    if not Path(full_temp_file_path).exists():
        shutil.copy2(file_path, full_temp_file_path)
    return full_temp_file_path