import functools
from enum import Enum
from urllib.parse import urlencode
Cacheable wrapper over urlencode

    Args:
        key (str): The key of the parameter to encode.
        value (str | bytes | Enum): The value of the parameter to encode.

    Returns:
        str: The encoded parameter.
    