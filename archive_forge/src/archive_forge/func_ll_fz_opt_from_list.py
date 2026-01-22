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
def ll_fz_opt_from_list(opt, optlist):
    """
    Low-level wrapper for `::fz_opt_from_list()`.
    Return the index of a (case-insensitive) option within an optlist.

    For instance for optlist = "Foo|Bar|Baz", and  opt = "bar",
    this would return 1.

    If the optlist ends with "|*" then that is a catch all case and
    matches all options allowing the caller to process it itself.
    fz_optarg will be set to point to the option, and the return
    value will be the index of the '*' option within that list.

    If an optlist entry ends with ':' (e.g. "Foo:") then that option
    may have suboptions appended to it (for example "JPG:80") and
    fz_optarg will be set to point at "80". Otherwise fz_optarg will
    be set to NULL.

    In the event of no-match found, prints an error and returns -1.
    """
    return _mupdf.ll_fz_opt_from_list(opt, optlist)