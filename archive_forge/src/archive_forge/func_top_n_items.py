from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
def top_n_items(self, n: int, sort: Optional[bool]=None) -> Dict[str, Union[int, float]]:
    """
        Gets the top n items
        """
    if sort:
        return dict(sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:n])
    return dict(list(self.data.items())[:n])