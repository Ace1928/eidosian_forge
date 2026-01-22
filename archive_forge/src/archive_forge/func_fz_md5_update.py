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
def fz_md5_update(self, input, inlen):
    """
        Class-aware wrapper for `::fz_md5_update()`.
        	MD5 block update operation. Continues an MD5 message-digest
        	operation, processing another message block, and updating the
        	context.

        	Never throws an exception.
        """
    return _mupdf.FzMd5_fz_md5_update(self, input, inlen)