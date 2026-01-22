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
def dataset_info(self, repo_id: str, *, revision: Optional[str]=None, timeout: Optional[float]=None, files_metadata: bool=False, token: Optional[Union[bool, str]]=None) -> DatasetInfo:
    """
        Get info on one specific dataset on huggingface.co.

        Dataset can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the dataset repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`hf_api.DatasetInfo`]: The dataset repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
    headers = self._build_hf_headers(token=token)
    path = f'{self.endpoint}/api/datasets/{repo_id}' if revision is None else f'{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}'
    params = {}
    if files_metadata:
        params['blobs'] = True
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
    hf_raise_for_status(r)
    data = r.json()
    return DatasetInfo(**data)