from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
def has_error_label(self, label: str) -> bool:
    """Return True if this error contains the given label.

        .. versionadded:: 3.7
        """
    return label in self._error_labels