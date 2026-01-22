from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def build_extension_item(name: ExtensionName, parameters: List[ExtensionParameter]) -> str:
    """
    Build an extension definition.

    This is the reverse of :func:`parse_extension_item`.

    """
    return '; '.join([cast(str, name)] + [name if value is None else f'{name}={value}' for name, value in parameters])