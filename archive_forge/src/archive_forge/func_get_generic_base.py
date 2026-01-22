import inspect
import platform
import sys
import threading
from collections.abc import Mapping, Sequence  # noqa: F401
from typing import _GenericAlias
def get_generic_base(cl):
    """If this is a generic class (A[str]), return the generic base for it."""
    if cl.__class__ is _GenericAlias:
        return cl.__origin__
    return None