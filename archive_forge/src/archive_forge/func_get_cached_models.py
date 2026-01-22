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
@_deprecate_method(version='4.39.0', message='This method is outdated and does not support the new cache system.')
def get_cached_models(cache_dir: Union[str, Path]=None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape `(model_url,
    etag, size_MB)`. Filenames in `cache_dir` are use to get the metadata for each model, only urls ending with *.bin*
    are added.

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []
    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith('.json'):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding='utf-8') as meta_file:
                metadata = json.load(meta_file)
                url = metadata['url']
                etag = metadata['etag']
                if url.endswith('.bin'):
                    size_MB = os.path.getsize(meta_path.strip('.json')) / 1000000.0
                    cached_models.append((url, etag, size_MB))
    return cached_models