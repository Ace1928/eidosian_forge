from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
@langPrefix.setter
def langPrefix(self, value: str) -> None:
    self._options['langPrefix'] = value