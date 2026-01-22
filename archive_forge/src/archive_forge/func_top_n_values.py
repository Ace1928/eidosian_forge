from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
def top_n_values(self, n: int, sort: Optional[bool]=None) -> List[Union[int, float]]:
    """
        Gets the top n values
        """
    if sort:
        return sorted(self.data.values(), key=lambda x: x, reverse=True)[:n]
    return list(self.data.values())[:n]