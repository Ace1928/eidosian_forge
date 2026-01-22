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
def objects_to_list_of_image_dicts(objs: Union[List[str], List[dict], List[np.ndarray], List['PIL.Image.Image']]) -> List[dict]:
    """Encode a list of objects into a format suitable for creating an extension array of type `ImageExtensionType`."""
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support encoding images, please install 'Pillow'.")
    if objs:
        _, obj = first_non_null_value(objs)
        if isinstance(obj, str):
            return [{'path': obj, 'bytes': None} if obj is not None else None for obj in objs]
        if isinstance(obj, np.ndarray):
            obj_to_image_dict_func = no_op_if_value_is_null(encode_np_array)
            return [obj_to_image_dict_func(obj) for obj in objs]
        elif isinstance(obj, PIL.Image.Image):
            obj_to_image_dict_func = no_op_if_value_is_null(encode_pil_image)
            return [obj_to_image_dict_func(obj) for obj in objs]
        else:
            return objs
    else:
        return objs