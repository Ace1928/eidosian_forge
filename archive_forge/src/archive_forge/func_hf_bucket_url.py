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
def hf_bucket_url(identifier: str, filename: str, use_cdn=False, dataset=True) -> str:
    if dataset:
        endpoint = config.CLOUDFRONT_DATASETS_DISTRIB_PREFIX if use_cdn else config.S3_DATASETS_BUCKET_PREFIX
    else:
        endpoint = config.CLOUDFRONT_METRICS_DISTRIB_PREFIX if use_cdn else config.S3_METRICS_BUCKET_PREFIX
    return '/'.join((endpoint, identifier, filename))