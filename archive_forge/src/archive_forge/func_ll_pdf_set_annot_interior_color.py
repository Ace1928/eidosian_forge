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
def ll_pdf_set_annot_interior_color(annot, color):
    """
    Low-level Python version of pdf_set_annot_color() using
    pdf_set_annot_color2().
    """
    if isinstance(color, float):
        ll_pdf_set_annot_interior_color2(annot, 1, color, 0, 0, 0)
    elif len(color) == 1:
        ll_pdf_set_annot_interior_color2(annot, 1, color[0], 0, 0, 0)
    elif len(color) == 2:
        ll_pdf_set_annot_interior_color2(annot, 2, color[0], color[1], 0, 0)
    elif len(color) == 3:
        ll_pdf_set_annot_interior_color2(annot, 3, color[0], color[1], color[2], 0)
    elif len(color) == 4:
        ll_pdf_set_annot_interior_color2(annot, 4, color[0], color[1], color[2], color[3])
    else:
        raise Exception(f'Unexpected color should be float or list of 1-4 floats: {color}')