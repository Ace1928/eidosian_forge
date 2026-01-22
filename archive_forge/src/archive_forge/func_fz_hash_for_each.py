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
def fz_hash_for_each(self, state, callback):
    """
        Class-aware wrapper for `::fz_hash_for_each()`.
        	Iterate over the entries in a hash table.
        """
    return _mupdf.FzHashTable_fz_hash_for_each(self, state, callback)