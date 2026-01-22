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
def list_rejected_access_requests(self, repo_id: str, *, repo_type: Optional[str]=None, token: Optional[str]=None) -> List[AccessRequest]:
    """
        Get rejected access requests for a given gated repo.

        A rejected request means the user has requested access to the repo and the request has been explicitly rejected
        by a repo owner (either you or another user from your organization). The user cannot download any file of the
        repo. Rejected requests can be accepted or cancelled at any time using [`accept_access_request`] and
        [`cancel_access_request`]. A cancelled request will go back to the pending list while an accepted request will
        go to the accepted list.

        For more info about gated repos, see https://huggingface.co/docs/hub/models-gated.

        Args:
            repo_id (`str`):
                The id of the repo to get access requests for.
            repo_type (`str`, *optional*):
                The type of the repo to get access requests for. Must be one of `model`, `dataset` or `space`.
                Defaults to `model`.
            token (`str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).

        Returns:
            `List[AccessRequest]`: A list of [`AccessRequest`] objects. Each time contains a `username`, `email`,
            `status` and `timestamp` attribute. If the gated repo has a custom form, the `fields` attribute will
            be populated with user's answers.

        Raises:
            `HTTPError`:
                HTTP 400 if the repo is not gated.
            `HTTPError`:
                HTTP 403 if you only have read-only access to the repo. This can be the case if you don't have `write`
                or `admin` role in the organization the repo belongs to or if you passed a `read` token.

        Example:
        ```py
        >>> from huggingface_hub import list_rejected_access_requests

        >>> requests = list_rejected_access_requests("meta-llama/Llama-2-7b")
        >>> len(requests)
        411
        >>> requests[0]
        [
            AccessRequest(
                username='clem',
                fullname='Clem 🤗',
                email='***',
                timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
                status='rejected',
                fields=None,
            ),
            ...
        ]
        ```
        """
    return self._list_access_requests(repo_id, 'rejected', repo_type=repo_type, token=token)