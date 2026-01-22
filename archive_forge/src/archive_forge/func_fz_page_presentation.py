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
def fz_page_presentation(self, transition, duration):
    """
        Class-aware wrapper for `::fz_page_presentation()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_page_presentation(::fz_transition *transition)` => `(fz_transition *, float duration)`

        	Get the presentation details for a given page.

        	transition: A pointer to a transition struct to fill out.

        	duration: A pointer to a place to set the page duration in
        	seconds. Will be set to 0 if no transition is specified for the
        	page.

        	Returns: a pointer to the transition structure, or NULL if there
        	is no transition specified for the page.
        """
    return _mupdf.FzPage_fz_page_presentation(self, transition, duration)