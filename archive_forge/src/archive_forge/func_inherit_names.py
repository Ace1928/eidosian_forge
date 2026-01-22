from __future__ import annotations
from typing import (
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.indexes.base import Index
def inherit_names(names: list[str], delegate: type, cache: bool=False, wrap: bool=False) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

    def wrapper(cls: type[_ExtensionIndexT]) -> type[_ExtensionIndexT]:
        for name in names:
            meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)
        return cls
    return wrapper