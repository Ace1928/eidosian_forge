from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
@property
def average_s(self) -> str:
    """
        Returns the average in seconds
        """
    return self.pformat_duration(self.average, short=1, include_ms=False)