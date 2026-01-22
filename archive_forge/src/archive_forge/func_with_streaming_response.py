from __future__ import annotations
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import maybe_transform
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .....pagination import SyncCursorPage, AsyncCursorPage
from ....._base_client import (
from .....types.beta.threads.runs import RunStep, step_list_params
@cached_property
def with_streaming_response(self) -> AsyncStepsWithStreamingResponse:
    return AsyncStepsWithStreamingResponse(self)