from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
@validate_hf_hub_args
def delete_branch(self, repo_id: str, *, branch: str, token: Optional[str]=None, repo_type: Optional[str]=None) -> None:
    """
        Delete a branch from a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a branch will be deleted.
                Example: `"user/my-cool-model"`.

            branch (`str`):
                The name of the branch to delete.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if creating a branch on a dataset or
                space, `None` or `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a protected branch. Ex: `main` cannot be deleted.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a branch that does not exist.

        """
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    branch = quote(branch, safe='')
    branch_url = f'{self.endpoint}/api/{repo_type}s/{repo_id}/branch/{branch}'
    headers = self._build_hf_headers(token=token, is_write_action=True)
    response = get_session().delete(url=branch_url, headers=headers)
    hf_raise_for_status(response)