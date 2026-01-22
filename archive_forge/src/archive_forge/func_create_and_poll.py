from __future__ import annotations
import time
import typing_extensions
from typing import Union, Iterable, Optional, overload
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
from .....types.beta import (
from ....._base_client import (
from .....lib.streaming import (
from .....types.beta.threads import (
def create_and_poll(self, *, assistant_id: str, additional_instructions: Optional[str] | NotGiven=NOT_GIVEN, additional_messages: Optional[Iterable[run_create_params.AdditionalMessage]] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, max_completion_tokens: Optional[int] | NotGiven=NOT_GIVEN, max_prompt_tokens: Optional[int] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Union[str, Literal['gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k-0613'], None] | NotGiven=NOT_GIVEN, response_format: Optional[AssistantResponseFormatOptionParam] | NotGiven=NOT_GIVEN, temperature: Optional[float] | NotGiven=NOT_GIVEN, tool_choice: Optional[AssistantToolChoiceOptionParam] | NotGiven=NOT_GIVEN, tools: Optional[Iterable[AssistantToolParam]] | NotGiven=NOT_GIVEN, top_p: Optional[float] | NotGiven=NOT_GIVEN, truncation_strategy: Optional[run_create_params.TruncationStrategy] | NotGiven=NOT_GIVEN, poll_interval_ms: int | NotGiven=NOT_GIVEN, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
    """
        A helper to create a run an poll for a terminal state. More information on Run
        lifecycles can be found here:
        https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps
        """
    run = self.create(thread_id=thread_id, assistant_id=assistant_id, additional_instructions=additional_instructions, additional_messages=additional_messages, instructions=instructions, max_completion_tokens=max_completion_tokens, max_prompt_tokens=max_prompt_tokens, metadata=metadata, model=model, response_format=response_format, temperature=temperature, tool_choice=tool_choice, stream=False, tools=tools, truncation_strategy=truncation_strategy, top_p=top_p, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout)
    return self.poll(run.id, thread_id=thread_id, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, poll_interval_ms=poll_interval_ms, timeout=timeout)