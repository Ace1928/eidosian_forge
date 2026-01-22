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
def cancel_access_request(self, repo_id: str, user: str, *, repo_type: Optional[str]=None, token: Optional[str]=None) -> None:
    """
        Cancel an access request from a user for a given gated repo.

        A cancelled request will go back to the pending list and the user will lose access to the repo.

        For more info about gated repos, see https://huggingface.co/docs/hub/models-gated.

        Args:
            repo_id (`str`):
                The id of the repo to cancel access request for.
            user (`str`):
                The username of the user which access request should be cancelled.
            repo_type (`str`, *optional*):
                The type of the repo to cancel access request for. Must be one of `model`, `dataset` or `space`.
                Defaults to `model`.
            token (`str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).

        Raises:
            `HTTPError`:
                HTTP 400 if the repo is not gated.
            `HTTPError`:
                HTTP 403 if you only have read-only access to the repo. This can be the case if you don't have `write`
                or `admin` role in the organization the repo belongs to or if you passed a `read` token.
            `HTTPError`:
                HTTP 404 if the user does not exist on the Hub.
            `HTTPError`:
                HTTP 404 if the user access request cannot be found.
            `HTTPError`:
                HTTP 404 if the user access request is already in the pending list.
        """
    self._handle_access_request(repo_id, user, 'pending', repo_type=repo_type, token=token)