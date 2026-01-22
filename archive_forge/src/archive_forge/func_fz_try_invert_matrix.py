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
def fz_try_invert_matrix(self, src):
    """
        Class-aware wrapper for `::fz_try_invert_matrix()`.
        	Attempt to create an inverse matrix.

        	inverse: Place to store inverse matrix.

        	matrix: Matrix to invert. A degenerate matrix, where the
        	determinant is equal to zero, can not be inverted.

        	Returns 1 if matrix is degenerate (singular), or 0 otherwise.
        """
    return _mupdf.FzMatrix_fz_try_invert_matrix(self, src)