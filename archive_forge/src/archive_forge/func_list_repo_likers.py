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
def list_repo_likers(self, repo_id: str, *, repo_type: Optional[str]=None, token: Optional[str]=None) -> List[User]:
    """
        List all users who liked a given repo on the hugging Face Hub.

        See also [`like`] and [`list_liked_repos`].

        Args:
            repo_id (`str`):
                The repository to retrieve . Example: `"user/my-cool-model"`.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns:
            `List[User]`: a list of [`User`] objects.
        """
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    path = f'{self.endpoint}/api/{repo_type}s/{repo_id}/likers'
    headers = self._build_hf_headers(token=token)
    response = get_session().get(path, headers=headers)
    hf_raise_for_status(response)
    likers_data = response.json()
    return [User(username=user_data['user'], fullname=user_data['fullname'], avatar_url=user_data['avatarUrl']) for user_data in likers_data]