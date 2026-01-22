from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple
def setprocessor(self, tags: str | tuple[str, ...], processor: _Processor) -> None:
    if isinstance(tags, str):
        tags = tuple(tags.split(':'))
    else:
        assert isinstance(tags, tuple)
    self._tags2proc[tags] = processor