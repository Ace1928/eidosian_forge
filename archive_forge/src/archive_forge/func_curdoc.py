from __future__ import annotations
import logging # isort:skip
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, cast
from ..document import Document
from .state import curstate
def curdoc() -> Document:
    """ Return the document for the current default state.

    Returns:
        Document : the current default document object.

    """
    if len(_PATCHED_CURDOCS) > 0:
        doc = _PATCHED_CURDOCS[-1]()
        if doc is None:
            raise RuntimeError('Patched curdoc has been previously destroyed')
        return cast(Document, doc)
    return curstate().document