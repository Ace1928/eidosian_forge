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
def ll_fz_new_hash_table(initialsize, keylen, lock, drop_val):
    """
    Low-level wrapper for `::fz_new_hash_table()`.
    Create a new hash table.

    initialsize: The initial size of the hashtable. The hashtable
    may grow (double in size) if it starts to get crowded (80%
    full).

    keylen: byte length for each key.

    lock: -1 for no lock, otherwise the FZ_LOCK to use to protect
    this table.

    drop_val: Function to use to destroy values on table drop.
    """
    return _mupdf.ll_fz_new_hash_table(initialsize, keylen, lock, drop_val)