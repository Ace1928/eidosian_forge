from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def dformat(self, duration: float, pretty: bool=None, short: int=None, include_ms: bool=None, as_int: bool=False) -> Dict[str, Union[float, int]]:
    """
        Formats a duration (secs) into a dict
        """
    pretty = pretty if pretty is not None else self.format_pretty
    short = short if short is not None else self.format_short
    include_ms = include_ms if include_ms is not None else self.format_ms
    return self.dformat_duration(duration, pretty=pretty, short=short, include_ms=include_ms, as_int=as_int)