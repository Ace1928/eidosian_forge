from __future__ import annotations
from typing import Generator
from typing import NamedTuple
from flake8.violation import Violation
def error_codes(self) -> list[str]:
    """Return all unique error codes stored.

        :returns:
            Sorted list of error codes.
        """
    return sorted({key.code for key in self._store})