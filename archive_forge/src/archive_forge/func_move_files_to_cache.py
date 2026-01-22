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
def move_files_to_cache(data: Any, block: Block, postprocess: bool=False, check_in_upload_folder=False, keep_in_cache=False) -> dict:
    """Move any files in `data` to cache and (optionally), adds URL prefixes (/file=...) needed to access the cached file.
    Also handles the case where the file is on an external Gradio app (/proxy=...).

    Runs after .postprocess() and before .preprocess().

    Args:
        data: The input or output data for a component. Can be a dictionary or a dataclass
        block: The component whose data is being processed
        postprocess: Whether its running from postprocessing
        check_in_upload_folder: If True, instead of moving the file to cache, checks if the file is in already in cache (exception if not).
        keep_in_cache: If True, the file will not be deleted from cache when the server is shut down.
    """

    def _move_to_cache(d: dict):
        payload = FileData(**d)
        if payload.url and postprocess and client_utils.is_http_url_like(payload.url):
            payload.path = payload.url
        elif utils.is_static_file(payload):
            pass
        elif not block.proxy_url:
            if check_in_upload_folder and (not client_utils.is_http_url_like(payload.path)):
                path = os.path.abspath(payload.path)
                if not is_in_or_equal(path, get_upload_folder()):
                    raise ValueError(f'File {path} is not in the upload folder and cannot be accessed.')
            if not payload.is_stream:
                temp_file_path = block.move_resource_to_block_cache(payload.path)
                if temp_file_path is None:
                    raise ValueError('Did not determine a file path for the resource.')
                payload.path = temp_file_path
                if keep_in_cache:
                    block.keep_in_cache.add(payload.path)
        url_prefix = '/stream/' if payload.is_stream else '/file='
        if block.proxy_url:
            proxy_url = block.proxy_url.rstrip('/')
            url = f'/proxy={proxy_url}{url_prefix}{payload.path}'
        elif client_utils.is_http_url_like(payload.path) or payload.path.startswith(f'{url_prefix}'):
            url = payload.path
        else:
            url = f'{url_prefix}{payload.path}'
        payload.url = url
        return payload.model_dump()
    if isinstance(data, (GradioRootModel, GradioModel)):
        data = data.model_dump()
    return client_utils.traverse(data, _move_to_cache, client_utils.is_file_obj)