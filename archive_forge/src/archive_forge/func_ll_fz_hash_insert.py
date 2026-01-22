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
def ll_fz_hash_insert(table, key, val):
    """
    Low-level wrapper for `::fz_hash_insert()`.
    Insert a new key/value pair into the hash table.

    If an existing entry with the same key is found, no change is
    made to the hash table, and a pointer to the existing value is
    returned.

    If no existing entry with the same key is found, ownership of
    val passes in, key is copied, and NULL is returned.
    """
    return _mupdf.ll_fz_hash_insert(table, key, val)