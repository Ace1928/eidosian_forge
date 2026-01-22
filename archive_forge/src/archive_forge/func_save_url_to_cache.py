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
def save_url_to_cache(url: str, cache_dir: str) -> str:
    """Downloads a file and makes a temporary file path for a copy if does not already
    exist. Otherwise returns the path to the existing temp file."""
    temp_dir = hash_url(url)
    temp_dir = Path(cache_dir) / temp_dir
    temp_dir.mkdir(exist_ok=True, parents=True)
    name = client_utils.strip_invalid_filename_characters(Path(url).name)
    full_temp_file_path = str(abspath(temp_dir / name))
    if not Path(full_temp_file_path).exists():
        with sync_client.stream('GET', url, follow_redirects=True) as r, open(full_temp_file_path, 'wb') as f:
            for chunk in r.iter_raw():
                f.write(chunk)
    return full_temp_file_path