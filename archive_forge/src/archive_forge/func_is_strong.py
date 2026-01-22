from __future__ import annotations
from collections.abc import Collection
def is_strong(self, etag):
    """Check if an etag is strong."""
    return etag in self._strong