from __future__ import annotations
from collections.abc import Collection
def contains_weak(self, etag):
    """Check if an etag is part of the set including weak and strong tags."""
    return self.is_weak(etag) or self.contains(etag)