import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def safe_contains(value, container):
    """ Perform "in" containment check, allowing for TypeErrors.

    This is required because in some circumstances ``x in y`` can raise a
    TypeError.  In these cases we make the (reasonable) assumption that the
    value is _not_ contained in the container.
    """
    if isinstance(container, enum.EnumMeta):
        if not isinstance(value, enum.Enum):
            return False
    try:
        return value in container
    except TypeError:
        return False