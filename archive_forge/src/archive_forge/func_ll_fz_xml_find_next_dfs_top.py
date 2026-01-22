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
def ll_fz_xml_find_next_dfs_top(item, tag, att, match, top):
    """
    Low-level wrapper for `::fz_xml_find_next_dfs_top()`.
    Perform a depth first search onwards from item, returning the first
    child that matches the given tag (or any tag if tag is NULL),
    with the given attribute (if att is non NULL), that matches
    match (if match is non NULL). The search stops if it ever reaches
    the top of the tree, or the declared 'top' item.
    """
    return _mupdf.ll_fz_xml_find_next_dfs_top(item, tag, att, match, top)