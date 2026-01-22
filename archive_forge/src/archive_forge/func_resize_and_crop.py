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
def resize_and_crop(img, size, crop_type='center'):
    """
    Resize and crop an image to fit the specified size.
    args:
        size: `(width, height)` tuple. Pass `None` for either width or height
        to only crop and resize the other.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'middle' or
            'bottom/right' of the image to fit the size.
    raises:
        ValueError: if an invalid `crop_type` is provided.
    """
    if crop_type == 'top':
        center = (0, 0)
    elif crop_type == 'center':
        center = (0.5, 0.5)
    else:
        raise ValueError
    resize = list(size)
    if size[0] is None:
        resize[0] = img.size[0]
    if size[1] is None:
        resize[1] = img.size[1]
    return ImageOps.fit(img, resize, centering=center)