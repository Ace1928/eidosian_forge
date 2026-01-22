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
def list_collections(self, *, owner: Union[List[str], str, None]=None, item: Union[List[str], str, None]=None, sort: Optional[Literal['lastModified', 'trending', 'upvotes']]=None, limit: Optional[int]=None, token: Optional[Union[bool, str]]=None) -> Iterable[Collection]:
    """List collections on the Huggingface Hub, given some filters.

        <Tip warning={true}>

        When listing collections, the item list per collection is truncated to 4 items maximum. To retrieve all items
        from a collection, you must use [`get_collection`].

        </Tip>

        Args:
            owner (`List[str]` or `str`, *optional*):
                Filter by owner's username.
            item (`List[str]` or `str`, *optional*):
                Filter collections containing a particular items. Example: `"models/teknium/OpenHermes-2.5-Mistral-7B"`, `"datasets/squad"` or `"papers/2311.12983"`.
            sort (`Literal["lastModified", "trending", "upvotes"]`, *optional*):
                Sort collections by last modified, trending or upvotes.
            limit (`int`, *optional*):
                Maximum number of collections to be returned.
            token (`bool` or `str`, *optional*):
                An authentication token (see https://huggingface.co/settings/token).

        Returns:
            `Iterable[Collection]`: an iterable of [`Collection`] objects.
        """
    path = f'{self.endpoint}/api/collections'
    headers = self._build_hf_headers(token=token)
    params: Dict = {}
    if owner is not None:
        params.update({'owner': owner})
    if item is not None:
        params.update({'item': item})
    if sort is not None:
        params.update({'sort': sort})
    if limit is not None:
        params.update({'limit': limit})
    items = paginate(path, headers=headers, params=params)
    if limit is not None:
        items = islice(items, limit)
    for position, collection_data in enumerate(items):
        yield Collection(position=position, **collection_data)