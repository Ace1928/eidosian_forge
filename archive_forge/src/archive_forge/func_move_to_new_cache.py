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
def move_to_new_cache(file, repo, filename, revision, etag, commit_hash):
    """
    Move file to repo following the new huggingface hub cache organization.
    """
    os.makedirs(repo, exist_ok=True)
    os.makedirs(os.path.join(repo, 'refs'), exist_ok=True)
    if revision != commit_hash:
        ref_path = os.path.join(repo, 'refs', revision)
        with open(ref_path, 'w') as f:
            f.write(commit_hash)
    os.makedirs(os.path.join(repo, 'blobs'), exist_ok=True)
    blob_path = os.path.join(repo, 'blobs', etag)
    shutil.move(file, blob_path)
    os.makedirs(os.path.join(repo, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(repo, 'snapshots', commit_hash), exist_ok=True)
    pointer_path = os.path.join(repo, 'snapshots', commit_hash, filename)
    huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
    clean_files_for(file)