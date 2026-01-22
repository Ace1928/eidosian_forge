import copy
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import time
import urllib
import warnings
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urljoin, urlparse
import fsspec
import huggingface_hub
import requests
from fsspec.core import strip_protocol
from fsspec.utils import can_be_local
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from .. import __version__, config
from ..download.download_config import DownloadConfig
from . import _tqdm, logging
from . import tqdm as hf_tqdm
from ._filelock import FileLock
from .extract import ExtractManager
def get_authentication_headers_for_url(url: str, token: Optional[Union[str, bool]]=None, use_auth_token: Optional[Union[str, bool]]='deprecated') -> dict:
    """Handle the HF authentication"""
    if use_auth_token != 'deprecated':
        warnings.warn(f"'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'token={use_auth_token}' instead.", FutureWarning)
        token = use_auth_token
    if url.startswith(config.HF_ENDPOINT):
        return huggingface_hub.utils.build_hf_headers(token=token, library_name='datasets', library_version=__version__)
    else:
        return {}