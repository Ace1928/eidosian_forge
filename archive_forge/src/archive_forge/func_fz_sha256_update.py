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
def fz_sha256_update(self, input, inlen):
    """
        Class-aware wrapper for `::fz_sha256_update()`.
        	SHA256 block update operation. Continues an SHA256 message-
        	digest operation, processing another message block, and updating
        	the context.

        	Never throws an exception.
        """
    return _mupdf.FzSha256_fz_sha256_update(self, input, inlen)