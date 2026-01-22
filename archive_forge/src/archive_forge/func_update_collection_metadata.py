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
def update_collection_metadata(self, collection_slug: str, *, title: Optional[str]=None, description: Optional[str]=None, position: Optional[int]=None, private: Optional[bool]=None, theme: Optional[str]=None, token: Optional[str]=None) -> Collection:
    """Update metadata of a collection on the Hub.

        All arguments are optional. Only provided metadata will be updated.

        Args:
            collection_slug (`str`):
                Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
            title (`str`):
                Title of the collection to update.
            description (`str`, *optional*):
                Description of the collection to update.
            position (`int`, *optional*):
                New position of the collection in the list of collections of the user.
            private (`bool`, *optional*):
                Whether the collection should be private or not.
            theme (`str`, *optional*):
                Theme of the collection on the Hub.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns: [`Collection`]

        Example:

        ```py
        >>> from huggingface_hub import update_collection_metadata
        >>> collection = update_collection_metadata(
        ...     collection_slug="username/iccv-2023-64f9a55bb3115b4f513ec026",
        ...     title="ICCV Oct. 2023"
        ...     description="Portfolio of models, datasets, papers and demos I presented at ICCV Oct. 2023",
        ...     private=False,
        ...     theme="pink",
        ... )
        >>> collection.slug
        "username/iccv-oct-2023-64f9a55bb3115b4f513ec026"
        # ^collection slug got updated but not the trailing ID
        ```
        """
    payload = {'position': position, 'private': private, 'theme': theme, 'title': title, 'description': description}
    r = get_session().patch(f'{self.endpoint}/api/collections/{collection_slug}', headers=self._build_hf_headers(token=token), json={key: value for key, value in payload.items() if value is not None})
    hf_raise_for_status(r)
    return Collection(**{**r.json()['data'], 'endpoint': self.endpoint})