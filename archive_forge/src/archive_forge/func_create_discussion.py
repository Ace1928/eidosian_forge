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
def create_discussion(self, repo_id: str, title: str, *, token: Optional[str]=None, description: Optional[str]=None, repo_type: Optional[str]=None, pull_request: bool=False) -> DiscussionWithDetails:
    """Creates a Discussion or Pull Request.

        Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            pull_request (`bool`, *optional*):
                Whether to create a Pull Request or discussion. If `True`, creates a Pull Request.
                If `False`, creates a discussion. Defaults to `False`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
    if repo_type not in REPO_TYPES:
        raise ValueError(f'Invalid repo type, must be one of {REPO_TYPES}')
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    if description is not None:
        description = description.strip()
    description = description if description else f'{('Pull Request' if pull_request else 'Discussion')} opened with the [huggingface_hub Python library](https://huggingface.co/docs/huggingface_hub)'
    headers = self._build_hf_headers(token=token, is_write_action=True)
    resp = get_session().post(f'{self.endpoint}/api/{repo_type}s/{repo_id}/discussions', json={'title': title.strip(), 'description': description, 'pullRequest': pull_request}, headers=headers)
    hf_raise_for_status(resp)
    num = resp.json()['num']
    return self.get_discussion_details(repo_id=repo_id, repo_type=repo_type, discussion_num=num, token=token)