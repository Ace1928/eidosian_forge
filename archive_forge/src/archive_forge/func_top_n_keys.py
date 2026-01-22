from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
def top_n_keys(self, n: int, sort: Optional[bool]=None) -> List[str]:
    """
        Gets the top n keys
        """
    if sort:
        return sorted(self.data.keys(), key=lambda x: x, reverse=True)[:n]
    return list(self.data.keys())[:n]