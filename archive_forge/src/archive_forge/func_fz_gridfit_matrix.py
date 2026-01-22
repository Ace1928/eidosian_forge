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
def fz_gridfit_matrix(as_tiled, m):
    """
    Class-aware wrapper for `::fz_gridfit_matrix()`.
    	Grid fit a matrix.

    	as_tiled = 0 => adjust the matrix so that the image of the unit
    	square completely covers any pixel that was touched by the
    	image of the unit square under the original matrix.

    	as_tiled = 1 => adjust the matrix so that the corners of the
    	image of the unit square align with the closest integer corner
    	of the image of the unit square under the original matrix.
    """
    return _mupdf.fz_gridfit_matrix(as_tiled, m)