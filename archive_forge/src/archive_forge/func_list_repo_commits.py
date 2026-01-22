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
def list_repo_commits(self, repo_id: str, *, repo_type: Optional[str]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, formatted: bool=False) -> List[GitCommitInfo]:
    """
        Get the list of commits of a given revision for a repo on the Hub.

        Commits are sorted by date (last commit first).

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if listing commits from a dataset or a Space, `None` or `"model"` if
                listing from a model. Default is `None`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            formatted (`bool`):
                Whether to return the HTML-formatted title and description of the commits. Defaults to False.

        Example:
        ```py
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()

        # Commits are sorted by date (last commit first)
        >>> initial_commit = api.list_repo_commits("gpt2")[-1]

        # Initial commit is always a system commit containing the `.gitattributes` file.
        >>> initial_commit
        GitCommitInfo(
            commit_id='9b865efde13a30c13e0a33e536cf3e4a5a9d71d8',
            authors=['system'],
            created_at=datetime.datetime(2019, 2, 18, 10, 36, 15, tzinfo=datetime.timezone.utc),
            title='initial commit',
            message='',
            formatted_title=None,
            formatted_message=None
        )

        # Create an empty branch by deriving from initial commit
        >>> api.create_branch("gpt2", "new_empty_branch", revision=initial_commit.commit_id)
        ```

        Returns:
            List[[`GitCommitInfo`]]: list of objects containing information about the commits for a repo on the Hub.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.
        """
    repo_type = repo_type or REPO_TYPE_MODEL
    revision = quote(revision, safe='') if revision is not None else DEFAULT_REVISION
    return [GitCommitInfo(commit_id=item['id'], authors=[author['user'] for author in item['authors']], created_at=parse_datetime(item['date']), title=item['title'], message=item['message'], formatted_title=item.get('formatted', {}).get('title'), formatted_message=item.get('formatted', {}).get('message')) for item in paginate(f'{self.endpoint}/api/{repo_type}s/{repo_id}/commits/{revision}', headers=self._build_hf_headers(token=token), params={'expand[]': 'formatted'} if formatted else {})]