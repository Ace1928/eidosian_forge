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
def hide_discussion_comment(self, repo_id: str, discussion_num: int, comment_id: str, *, token: Optional[str]=None, repo_type: Optional[str]=None) -> DiscussionComment:
    """Hides a comment on a Discussion / Pull Request.

        <Tip warning={true}>
        Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.
        </Tip>

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the hidden comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
    warnings.warn("Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.", UserWarning)
    resp = self._post_discussion_changes(repo_id=repo_id, repo_type=repo_type, discussion_num=discussion_num, token=token, resource=f'comment/{comment_id.lower()}/hide')
    return deserialize_event(resp.json()['updatedComment'])