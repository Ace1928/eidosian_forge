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
def future_compatible(fn: CallableT) -> CallableT:
    """Wrap a method of `HfApi` to handle `run_as_future=True`.

    A method flagged as "future_compatible" will be called in a thread if `run_as_future=True` and return a
    `concurrent.futures.Future` instance. Otherwise, it will be called normally and return the result.
    """
    sig = inspect.signature(fn)
    args_params = list(sig.parameters)[1:]

    @wraps(fn)
    def _inner(self, *args, **kwargs):
        if 'run_as_future' in kwargs:
            run_as_future = kwargs['run_as_future']
            kwargs['run_as_future'] = False
        else:
            run_as_future = False
            for param, value in zip(args_params, args):
                if param == 'run_as_future':
                    run_as_future = value
                    break
        if run_as_future:
            return self.run_as_future(fn, self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    _inner.is_future_compatible = True
    return _inner