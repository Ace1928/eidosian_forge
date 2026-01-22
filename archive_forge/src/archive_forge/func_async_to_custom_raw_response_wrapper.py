from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from types import TracebackType
from typing import (
from typing_extensions import Awaitable, ParamSpec, override, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type, extract_type_var_from_base
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER, OVERRIDE_CAST_TO_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import OpenAIError, APIResponseValidationError
def async_to_custom_raw_response_wrapper(func: Callable[P, Awaitable[object]], response_cls: type[_AsyncAPIResponseT]) -> Callable[P, Awaitable[_AsyncAPIResponseT]]:
    """Higher order function that takes one of our bound API methods and an `APIResponse` class
    and wraps the method to support returning the given response class directly.

    Note: the given `response_cls` *must* be concrete, e.g. `class BinaryAPIResponse(APIResponse[bytes])`
    """

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Awaitable[_AsyncAPIResponseT]:
        extra_headers: dict[str, Any] = {**(cast(Any, kwargs.get('extra_headers')) or {})}
        extra_headers[RAW_RESPONSE_HEADER] = 'raw'
        extra_headers[OVERRIDE_CAST_TO_HEADER] = response_cls
        kwargs['extra_headers'] = extra_headers
        return cast(Awaitable[_AsyncAPIResponseT], func(*args, **kwargs))
    return wrapped