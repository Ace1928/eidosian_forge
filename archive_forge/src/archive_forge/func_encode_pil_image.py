import os
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import numpy as np
import pyarrow as pa
from .. import config
from ..download.download_config import DownloadConfig
from ..download.streaming_download_manager import xopen
from ..table import array_cast
from ..utils.file_utils import is_local_path
from ..utils.py_utils import first_non_null_value, no_op_if_value_is_null, string_to_dict
def encode_pil_image(image: 'PIL.Image.Image') -> dict:
    if hasattr(image, 'filename') and image.filename != '':
        return {'path': image.filename, 'bytes': None}
    else:
        return {'path': None, 'bytes': image_to_bytes(image)}