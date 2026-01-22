import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def rename_kwargs(func_name: str, kwargs: Dict[str, Any], aliases: Dict[str, str], fail: bool=False) -> None:
    """
    Helper function to deprecate arguments.

    Args:
        func_name: Name of the function to be deprecated
        kwargs:
        aliases:
        fail:
    """
    for old_term, new_term in aliases.items():
        if old_term in kwargs:
            if fail:
                raise DeprecationError(f'{old_term} is deprecated as an argument. Use {new_term} instead')
            if new_term in kwargs:
                raise TypeError(f'{func_name} received both {old_term} and {new_term} as an argument. {old_term} is deprecated. Use {new_term} instead.')
            kwargs[new_term] = kwargs.pop(old_term)
            warnings.warn(message=f'{old_term} is deprecated as an argument. Use {new_term} instead', category=DeprecationWarning)