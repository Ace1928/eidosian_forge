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
def fz_open_predict(self, predictor, columns, colors, bpc):
    """
        Class-aware wrapper for `::fz_open_predict()`.
        	predict filter performs pixel prediction on data read from
        	the chained filter.

        	predictor: 1 = copy, 2 = tiff, other = inline PNG predictor

        	columns: width of image in pixels

        	colors: number of components.

        	bpc: bits per component (typically 8)
        """
    return _mupdf.FzStream_fz_open_predict(self, predictor, columns, colors, bpc)