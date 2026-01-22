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
def fz_strsep(stringp, delim):
    """
    Class-aware wrapper for `::fz_strsep()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_strsep(const char *delim)` => `(char *, char *stringp)`

    	Given a pointer to a C string (or a pointer to NULL) break
    	it at the first occurrence of a delimiter char (from a given
    	set).

    	stringp: Pointer to a C string pointer (or NULL). Updated on
    	exit to point to the first char of the string after the
    	delimiter that was found. The string pointed to by stringp will
    	be corrupted by this call (as the found delimiter will be
    	overwritten by 0).

    	delim: A C string of acceptable delimiter characters.

    	Returns a pointer to a C string containing the chars of stringp
    	up to the first delimiter char (or the end of the string), or
    	NULL.
    """
    return _mupdf.fz_strsep(stringp, delim)