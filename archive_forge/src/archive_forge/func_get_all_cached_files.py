import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def get_all_cached_files(cache_dir=None):
    """
    Returns a list for all files cached with appropriate metadata.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []
    cached_files = []
    for file in os.listdir(cache_dir):
        meta_path = os.path.join(cache_dir, f'{file}.json')
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, encoding='utf-8') as meta_file:
            metadata = json.load(meta_file)
            url = metadata['url']
            etag = metadata['etag'].replace('"', '')
            cached_files.append({'file': file, 'url': url, 'etag': etag})
    return cached_files