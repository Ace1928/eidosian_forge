import sys
from ctypes.util import find_library
from . import constants
from .ffi import ffi
from .surfaces import (  # noqa isort:skip
from .patterns import (  # noqa isort:skip
from .fonts import (  # noqa isort:skip
from .context import Context  # noqa isort:skip
from .matrix import Matrix  # noqa isort:skip
from .constants import *  # noqa isort:skip
def dlopen(ffi, library_names, filenames):
    """Try various names for the same library, for different platforms."""
    exceptions = []
    for library_name in library_names:
        library_filename = find_library(library_name)
        if library_filename:
            filenames = (library_filename,) + filenames
        else:
            exceptions.append('no library called "{}" was found'.format(library_name))
    for filename in filenames:
        try:
            return ffi.dlopen(filename)
        except OSError as exception:
            exceptions.append(exception)
    error_message = '\n'.join((str(exception) for exception in exceptions))
    raise OSError(error_message)