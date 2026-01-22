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
def fz_dom_find(self, tag, att, match):
    """
        Class-aware wrapper for `::fz_dom_find()`.
        	Find the first element matching the requirements in a depth first traversal from elt.

        	The tagname must match tag, unless tag is NULL, when all tag names are considered to match.

        	If att is NULL, then all tags match.
        	Otherwise:
        		If match is NULL, then only nodes that have an att attribute match.
        		If match is non-NULL, then only nodes that have an att attribute that matches match match.

        	Returns NULL (if no match found), or a borrowed reference to the first matching element.
        """
    return _mupdf.FzXml_fz_dom_find(self, tag, att, match)