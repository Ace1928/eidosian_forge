import copy
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from urllib.parse import quote, unquote
import fsspec
from requests import Response
from ._commit_api import CommitOperationCopy, CommitOperationDelete
from .constants import DEFAULT_REVISION, ENDPOINT, REPO_TYPE_MODEL, REPO_TYPES_MAPPING, REPO_TYPES_URL_PREFIXES
from .file_download import hf_hub_url
from .hf_api import HfApi, LastCommitInfo, RepoFile
from .utils import (
def unresolve(self) -> str:
    repo_path = REPO_TYPES_URL_PREFIXES.get(self.repo_type, '') + self.repo_id
    if self._raw_revision:
        return f'{repo_path}@{self._raw_revision}/{self.path_in_repo}'.rstrip('/')
    elif self.revision != DEFAULT_REVISION:
        return f'{repo_path}@{safe_revision(self.revision)}/{self.path_in_repo}'.rstrip('/')
    else:
        return f'{repo_path}/{self.path_in_repo}'.rstrip('/')