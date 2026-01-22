from __future__ import annotations
import logging # isort:skip
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, cast
from ..document import Document
from .state import curstate
@contextmanager
def patch_curdoc(doc: Document | UnlockedDocumentProxy) -> Iterator[None]:
    """ Temporarily override the value of ``curdoc()`` and then return it to
    its original state.

    This context manager is useful for controlling the value of ``curdoc()``
    while invoking functions (e.g. callbacks). The cont

    Args:
        doc (Document) : new Document to use for ``curdoc()``

    """
    global _PATCHED_CURDOCS
    _PATCHED_CURDOCS.append(weakref.ref(doc))
    del doc
    yield
    _PATCHED_CURDOCS.pop()