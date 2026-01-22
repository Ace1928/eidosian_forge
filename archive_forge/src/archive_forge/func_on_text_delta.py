from __future__ import annotations
import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Callable, Iterable, Iterator, cast
from typing_extensions import Awaitable, AsyncIterable, AsyncIterator, assert_never
import httpx
from ..._utils import is_dict, is_list, consume_sync_iterator, consume_async_iterator
from ..._models import construct_type
from ..._streaming import Stream, AsyncStream
from ...types.beta import AssistantStreamEvent
from ...types.beta.threads import (
from ...types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta
def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
    """Callback that is fired whenever a text content delta is returned
        by the API.

        The first argument is just the delta as sent by the API and the second argument
        is the accumulated snapshot of the text. For example:

        on_text_delta(TextDelta(value="The"), Text(value="The")),
        on_text_delta(TextDelta(value=" solution"), Text(value="The solution")),
        on_text_delta(TextDelta(value=" to"), Text(value="The solution to")),
        on_text_delta(TextDelta(value=" the"), Text(value="The solution to the")),
        on_text_delta(TextDelta(value=" equation"), Text(value="The solution to the equivalent")),
        """