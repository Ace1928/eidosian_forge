from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_hash_remove(self, key):
    """
        Class-aware wrapper for `::fz_hash_remove()`.
        	Remove the entry for a given key.

        	The value is NOT freed, so the caller is expected to take care
        	of this.
        """
    return _mupdf.FzHashTable_fz_hash_remove(self, key)