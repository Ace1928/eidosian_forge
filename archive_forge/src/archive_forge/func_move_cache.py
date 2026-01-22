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
def move_cache(cache_dir=None, new_cache_dir=None, token=None):
    if new_cache_dir is None:
        new_cache_dir = TRANSFORMERS_CACHE
    if cache_dir is None:
        old_cache = Path(TRANSFORMERS_CACHE).parent / 'transformers'
        if os.path.isdir(str(old_cache)):
            cache_dir = str(old_cache)
        else:
            cache_dir = new_cache_dir
    cached_files = get_all_cached_files(cache_dir=cache_dir)
    logger.info(f'Moving {len(cached_files)} files to the new cache system')
    hub_metadata = {}
    for file_info in tqdm(cached_files):
        url = file_info.pop('url')
        if url not in hub_metadata:
            try:
                hub_metadata[url] = get_hf_file_metadata(url, token=token)
            except requests.HTTPError:
                continue
        etag, commit_hash = (hub_metadata[url].etag, hub_metadata[url].commit_hash)
        if etag is None or commit_hash is None:
            continue
        if file_info['etag'] != etag:
            clean_files_for(os.path.join(cache_dir, file_info['file']))
            continue
        url_info = extract_info_from_url(url)
        if url_info is None:
            continue
        repo = os.path.join(new_cache_dir, url_info['repo'])
        move_to_new_cache(file=os.path.join(cache_dir, file_info['file']), repo=repo, filename=url_info['filename'], revision=url_info['revision'], etag=etag, commit_hash=commit_hash)