import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec, overload
def rank_prefixed_message(message: str, rank: Optional[int]) -> str:
    """Add a prefix with the rank to a message."""
    if rank is not None:
        return f'[rank: {rank}] {message}'
    return message