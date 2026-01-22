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
def whoami(self, token: Optional[str]=None) -> Dict:
    """
        Call HF API to know "whoami".

        Args:
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
        """
    r = get_session().get(f'{self.endpoint}/api/whoami-v2', headers=self._build_hf_headers(token=token or self.token or True))
    try:
        hf_raise_for_status(r)
    except HTTPError as e:
        raise HTTPError("Invalid user token. If you didn't pass a user token, make sure you are properly logged in by executing `huggingface-cli login`, and if you did pass a user token, double-check it's correct.", request=e.request, response=e.response) from e
    return r.json()