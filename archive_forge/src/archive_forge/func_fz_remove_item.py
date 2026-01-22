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
def fz_remove_item(drop, key, type):
    """
    Class-aware wrapper for `::fz_remove_item()`.
    	Remove an item from the store.

    	If an item indexed by the given key exists in the store, remove
    	it.

    	drop: The function used to free the value (to ensure we get a
    	value of the correct type).

    	key: The key used to find the item to remove.

    	type: Functions used to manipulate the key.
    """
    return _mupdf.fz_remove_item(drop, key, type)