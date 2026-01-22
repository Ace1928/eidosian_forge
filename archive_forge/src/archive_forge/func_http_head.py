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
def http_head(url, proxies=None, headers=None, cookies=None, allow_redirects=True, timeout=10.0, max_retries=0) -> requests.Response:
    headers = copy.deepcopy(headers) or {}
    headers['user-agent'] = get_datasets_user_agent(user_agent=headers.get('user-agent'))
    response = _request_with_retry(method='HEAD', url=url, proxies=proxies, headers=headers, cookies=cookies, allow_redirects=allow_redirects, timeout=timeout, max_retries=max_retries)
    return response