from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def pattern_str_validator(__input_value: Any) -> typing.Pattern[str]:
    if isinstance(__input_value, typing.Pattern):
        if isinstance(__input_value.pattern, str):
            return __input_value
        else:
            raise PydanticCustomError('pattern_str_type', 'Input should be a string pattern')
    elif isinstance(__input_value, str):
        return compile_pattern(__input_value)
    elif isinstance(__input_value, bytes):
        raise PydanticCustomError('pattern_str_type', 'Input should be a string pattern')
    else:
        raise PydanticCustomError('pattern_type', 'Input should be a valid pattern')