import os
import sys
import stat
import fnmatch
import collections
import errno
def unregister_unpack_format(name):
    """Removes the pack format from the registry."""
    del _UNPACK_FORMATS[name]