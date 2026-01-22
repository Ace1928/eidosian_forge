import atexit
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from .hf_api import HfApi, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import (
from .utils._deprecation import _deprecate_method
def git_head_commit_url(self) -> str:
    """
        Get URL to last commit on HEAD. We assume it's been pushed, and the url
        scheme is the same one as for GitHub or HuggingFace.

        Returns:
            `str`: The URL to the current checked-out commit.
        """
    sha = self.git_head_hash()
    url = self.git_remote_url()
    if url.endswith('/'):
        url = url[:-1]
    return f'{url}/commit/{sha}'