from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def wrap_serializer_function_ser_schema(function: WrapSerializerFunction, *, is_field_serializer: bool | None=None, info_arg: bool | None=None, schema: CoreSchema | None=None, return_schema: CoreSchema | None=None, when_used: WhenUsed='always') -> WrapSerializerFunctionSerSchema:
    """
    Returns a schema for serialization with a wrap function, can be either a "general" or "field" function.

    Args:
        function: The function to use for serialization
        is_field_serializer: Whether the serializer is for a field, e.g. takes `model` as the first argument,
            and `info` includes `field_name`
        info_arg: Whether the function takes an `__info` argument
        schema: The schema to use for the inner serialization
        return_schema: Schema to use for serializing return value
        when_used: When the function should be called
    """
    if when_used == 'always':
        when_used = None
    return _dict_not_none(type='function-wrap', function=function, is_field_serializer=is_field_serializer, info_arg=info_arg, schema=schema, return_schema=return_schema, when_used=when_used)