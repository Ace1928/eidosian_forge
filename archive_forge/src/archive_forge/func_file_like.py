from __future__ import annotations
import io
import typing as ty
from copy import copy
from .openers import ImageOpener
@property
def file_like(self) -> str | io.IOBase | None:
    """Return ``self.fileobj`` if not None, otherwise ``self.filename``"""
    return self.fileobj if self.fileobj is not None else self.filename