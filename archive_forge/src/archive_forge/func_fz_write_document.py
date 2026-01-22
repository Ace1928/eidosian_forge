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
def fz_write_document(self, doc):
    """
        Class-aware wrapper for `::fz_write_document()`.
        	Convenience function to feed all the pages of a document to
        	fz_begin_page/fz_run_page/fz_end_page.
        """
    return _mupdf.FzDocumentWriter_fz_write_document(self, doc)