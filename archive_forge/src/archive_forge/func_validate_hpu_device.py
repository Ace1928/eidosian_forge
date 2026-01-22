import difflib
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from ._utils import _import_dotted_name
from torch._sources import get_source_lines_and_file
from torch.types import Storage
from torch.storage import _get_dtype_from_pickle_storage_type
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO
from typing_extensions import TypeAlias  # Python 3.10+
import copyreg
import pickle
import pathlib
import torch._weights_only_unpickler as _weights_only_unpickler
def validate_hpu_device(location):
    hpu = getattr(torch, 'hpu', None)
    assert hpu is not None, 'HPU device module is not loaded'
    device = hpu._utils._get_device_index(location, optional=True)
    if not hpu.is_available():
        raise RuntimeError("Attempting to deserialize object on a HPU device but torch.hpu.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.")
    device_count = hpu.device_count()
    if device >= device_count:
        raise RuntimeError(f'Attempting to deserialize object on HPU device {device} but torch.hpu.device_count() is {device_count}. Please use torch.load with map_location to map your storages to an existing device.')
    return device