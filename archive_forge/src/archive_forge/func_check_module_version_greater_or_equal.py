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
def check_module_version_greater_or_equal(module, req_version_tuple, error_if_malformed=True):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        version_strs = module.__version__.split('.')
        module_version = tuple((type(req_field)(version_strs[idx]) for idx, req_field in enumerate(req_version_tuple)))
        requirement_is_met = module_version >= req_version_tuple
    except Exception as e:
        message = f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared with tuple {str(req_version_tuple)}"
        if error_if_malformed:
            raise RuntimeError(message) from e
        else:
            warnings.warn(message + ', but continuing assuming that requirement is met')
            requirement_is_met = True
    return requirement_is_met