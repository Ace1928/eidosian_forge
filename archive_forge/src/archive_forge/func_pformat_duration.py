from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
@classmethod
def pformat_duration(cls, duration: float, pretty: bool=True, short: int=0, include_ms: bool=False) -> str:
    """
        Formats a duration (secs) into a string

        535003.0 -> 5 days, 5 hours, 50 minutes, 3 seconds
        3593.0 -> 59 minutes, 53 seconds
        """
    data = cls.dformat_duration(duration=duration, pretty=pretty, short=short, include_ms=include_ms, as_int=True)
    if not data:
        return '0 secs'
    sep = '' if short > 1 else ' '
    if short > 2:
        return ''.join([f'{v}{sep}{k}' for k, v in data.items()])
    return ' '.join([f'{v}{sep}{k}' for k, v in data.items()]) if short else ', '.join([f'{v}{sep}{k}' for k, v in data.items()])