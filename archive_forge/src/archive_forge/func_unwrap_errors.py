from __future__ import print_function, unicode_literals
import sys
import typing
import errno
import platform
from contextlib import contextmanager
from six import reraise
from . import errors
@contextmanager
def unwrap_errors(path_replace):
    """Get a context to map OS errors to their `fs.errors` counterpart.

    The context will re-write the paths in resource exceptions to be
    in the same context as the wrapped filesystem.

    The only parameter may be the path from the parent, if only one path
    is to be unwrapped. Or it may be a dictionary that maps wrapped
    paths on to unwrapped paths.

    """
    try:
        yield
    except errors.ResourceError as e:
        if hasattr(e, 'path'):
            if isinstance(path_replace, Mapping):
                e.path = path_replace.get(e.path, e.path)
            else:
                e.path = path_replace
        raise