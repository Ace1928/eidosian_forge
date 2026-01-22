from __future__ import annotations
import time
import typing_extensions
from typing import Mapping, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import FileObject, FileDeleted, file_list_params, file_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
from ..pagination import SyncPage, AsyncPage
from .._base_client import (
def wait_for_processing(self, id: str, *, poll_interval: float=5.0, max_wait_seconds: float=30 * 60) -> FileObject:
    """Waits for the given file to be processed, default timeout is 30 mins."""
    TERMINAL_STATES = {'processed', 'error', 'deleted'}
    start = time.time()
    file = self.retrieve(id)
    while file.status not in TERMINAL_STATES:
        self._sleep(poll_interval)
        file = self.retrieve(id)
        if time.time() - start > max_wait_seconds:
            raise RuntimeError(f'Giving up on waiting for file {id} to finish processing after {max_wait_seconds} seconds.')
    return file