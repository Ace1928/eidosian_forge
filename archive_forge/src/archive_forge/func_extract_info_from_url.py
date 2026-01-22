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
def extract_info_from_url(url):
    """
    Extract repo_name, revision and filename from an url.
    """
    search = re.search('^https://huggingface\\.co/(.*)/resolve/([^/]*)/(.*)$', url)
    if search is None:
        return None
    repo, revision, filename = search.groups()
    cache_repo = '--'.join(['models'] + repo.split('/'))
    return {'repo': cache_repo, 'revision': revision, 'filename': filename}