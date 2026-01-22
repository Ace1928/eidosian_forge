from __future__ import annotations
from typing import Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .runs import (
from .messages import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from .runs.runs import Runs, AsyncRuns
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...._streaming import Stream, AsyncStream
from ....types.beta import (
from ...._base_client import (
from ....lib.streaming import (
from .messages.messages import Messages, AsyncMessages
from ....types.beta.threads import Run
def create_and_run_stream(self, *, assistant_id: str, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Optional[str] | NotGiven=NOT_GIVEN, thread: thread_create_and_run_params.Thread | NotGiven=NOT_GIVEN, tools: Optional[Iterable[thread_create_and_run_params.Tool]] | NotGiven=NOT_GIVEN, event_handler: AsyncAssistantEventHandlerT | None=None, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler] | AsyncAssistantStreamManager[AsyncAssistantEventHandlerT]:
    """Create a thread and stream the run back"""
    extra_headers = {'OpenAI-Beta': 'assistants=v1', 'X-Stainless-Stream-Helper': 'threads.create_and_run_stream', 'X-Stainless-Custom-Event-Handler': 'true' if event_handler else 'false', **(extra_headers or {})}
    request = self._post('/threads/runs', body=maybe_transform({'assistant_id': assistant_id, 'instructions': instructions, 'metadata': metadata, 'model': model, 'stream': True, 'thread': thread, 'tools': tools}, thread_create_and_run_params.ThreadCreateAndRunParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run, stream=True, stream_cls=AsyncStream[AssistantStreamEvent])
    return AsyncAssistantStreamManager(request, event_handler=event_handler or AsyncAssistantEventHandler())