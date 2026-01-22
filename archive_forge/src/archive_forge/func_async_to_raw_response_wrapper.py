from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, Iterator, AsyncIterator, cast, overload
from typing_extensions import Awaitable, ParamSpec, override, deprecated, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import APIResponseValidationError
def async_to_raw_response_wrapper(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[LegacyAPIResponse[R]]]:
    """Higher order function that takes one of our bound API methods and wraps it
    to support returning the raw `APIResponse` object directly.
    """

    @functools.wraps(func)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> LegacyAPIResponse[R]:
        extra_headers: dict[str, str] = {**(cast(Any, kwargs.get('extra_headers')) or {})}
        extra_headers[RAW_RESPONSE_HEADER] = 'true'
        kwargs['extra_headers'] = extra_headers
        return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
    return wrapped