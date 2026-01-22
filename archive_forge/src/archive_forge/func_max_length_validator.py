from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def max_length_validator(x: Any, max_length: Any) -> Any:
    if len(x) > max_length:
        raise PydanticKnownError('too_long', {'field_type': 'Value', 'max_length': max_length, 'actual_length': len(x)})
    return x