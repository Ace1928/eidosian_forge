from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
def pickle_register(obj: Any) -> None:
    """Allow object to be pickled."""
    copyreg.pickle(obj, _pickle)