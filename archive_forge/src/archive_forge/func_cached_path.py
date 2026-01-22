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
def cached_path(url_or_filename, download_config=None, **download_kwargs) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
        ValueError: if it couldn't parse the url or filename correctly
        requests.exceptions.ConnectionError: in case of internet connection issue
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    cache_dir = download_config.cache_dir or config.DOWNLOADED_DATASETS_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if can_be_local(url_or_filename):
        url_or_filename = strip_protocol(url_or_filename)
    if is_remote_url(url_or_filename):
        output_path = get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=download_config.force_download, proxies=download_config.proxies, resume_download=download_config.resume_download, user_agent=download_config.user_agent, local_files_only=download_config.local_files_only, use_etag=download_config.use_etag, max_retries=download_config.max_retries, token=download_config.token, ignore_url_params=download_config.ignore_url_params, storage_options=download_config.storage_options, download_desc=download_config.download_desc)
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif is_local_path(url_or_filename):
        raise FileNotFoundError(f"Local file {url_or_filename} doesn't exist")
    else:
        raise ValueError(f'unable to parse {url_or_filename} as a URL or as a local path')
    if output_path is None:
        return output_path
    if download_config.extract_compressed_file:
        output_path = ExtractManager(cache_dir=download_config.cache_dir).extract(output_path, force_extract=download_config.force_extract)
    return relative_to_absolute_path(output_path)