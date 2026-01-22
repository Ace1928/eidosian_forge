import copy
import fnmatch
import inspect
import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, Literal, Optional, Tuple, Union
from urllib.parse import quote, urlparse
import requests
from filelock import FileLock
from huggingface_hub import constants
from . import __version__  # noqa: F401 # for backward compatibility
from .constants import (
from .utils import (
from .utils._deprecation import _deprecate_method
from .utils._headers import _http_user_agent
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility
from .utils._typing import HTTP_METHOD_T
from .utils.insecure_hashlib import sha256
def filename_to_url(filename, cache_dir: Optional[str]=None, legacy_cache_layout: bool=False) -> Tuple[str, str]:
    """
    Return the url and etag (which may be `None`) stored for `filename`. Raise
    `EnvironmentError` if `filename` or its stored metadata do not exist.

    Args:
        filename (`str`):
            The name of the file
        cache_dir (`str`, *optional*):
            The cache directory to use instead of the default one.
        legacy_cache_layout (`bool`, *optional*, defaults to `False`):
            If `True`, uses the legacy file cache layout i.e. just call `hf_hub_url`
            then `cached_download`. This is deprecated as the new cache layout is
            more powerful.
    """
    if not legacy_cache_layout:
        warnings.warn('`filename_to_url` uses the legacy way cache file layout', FutureWarning)
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError(f'file {cache_path} not found')
    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError(f'file {meta_path} not found')
    with open(meta_path, encoding='utf-8') as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']
    return (url, etag)