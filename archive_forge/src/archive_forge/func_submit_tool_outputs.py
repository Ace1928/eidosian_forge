from __future__ import annotations
from typing import Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from .steps import (
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import (
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....._streaming import Stream, AsyncStream
from .....pagination import SyncCursorPage, AsyncCursorPage
from .....types.beta import AssistantToolParam, AssistantStreamEvent
from ....._base_client import (
from .....lib.streaming import (
from .....types.beta.threads import (
@required_args(['thread_id', 'tool_outputs'], ['thread_id', 'stream', 'tool_outputs'])
def submit_tool_outputs(self, run_id: str, *, thread_id: str, tool_outputs: Iterable[run_submit_tool_outputs_params.ToolOutput], stream: Optional[Literal[False]] | Literal[True] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run | Stream[AssistantStreamEvent]:
    if not thread_id:
        raise ValueError(f'Expected a non-empty value for `thread_id` but received {thread_id!r}')
    if not run_id:
        raise ValueError(f'Expected a non-empty value for `run_id` but received {run_id!r}')
    extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
    return self._post(f'/threads/{thread_id}/runs/{run_id}/submit_tool_outputs', body=maybe_transform({'tool_outputs': tool_outputs, 'stream': stream}, run_submit_tool_outputs_params.RunSubmitToolOutputsParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run, stream=stream or False, stream_cls=Stream[AssistantStreamEvent])