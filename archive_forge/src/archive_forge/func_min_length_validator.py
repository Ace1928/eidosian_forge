from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def min_length_validator(x: Any, min_length: Any) -> Any:
    if not len(x) >= min_length:
        raise PydanticKnownError('too_short', {'field_type': 'Value', 'min_length': min_length, 'actual_length': len(x)})
    return x