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
def delete_inference_endpoint(self, name: str, *, namespace: Optional[str]=None, token: Optional[str]=None) -> None:
    """Delete an Inference Endpoint.

        This operation is not reversible. If you don't want to be charged for an Inference Endpoint, it is preferable
        to pause it with [`pause_inference_endpoint`] or scale it to zero with [`scale_to_zero_inference_endpoint`].

        For convenience, you can also delete an Inference Endpoint using [`InferenceEndpoint.delete`].

        Args:
            name (`str`):
                The name of the Inference Endpoint to delete.
            namespace (`str`, *optional*):
                The namespace in which the Inference Endpoint is located. Defaults to the current user.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).
        """
    namespace = namespace or self._get_namespace(token=token)
    response = get_session().delete(f'{INFERENCE_ENDPOINTS_ENDPOINT}/endpoint/{namespace}/{name}', headers=self._build_hf_headers(token=token))
    hf_raise_for_status(response)