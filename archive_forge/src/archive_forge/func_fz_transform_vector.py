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
def fz_transform_vector(self, *args):
    """
        *Overload 1:*
         Class-aware wrapper for `::fz_transform_vector()`.
        		Apply a transformation to a vector.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale and fz_rotate for how to create a matrix. Any
        		translation will be ignored.

        		vector: Pointer to vector to update.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_transform_vector()`.
        		Apply a transformation to a vector.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale and fz_rotate for how to create a matrix. Any
        		translation will be ignored.

        		vector: Pointer to vector to update.
        """
    return _mupdf.FzPoint_fz_transform_vector(self, *args)