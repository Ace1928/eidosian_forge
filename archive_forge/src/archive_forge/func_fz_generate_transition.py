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
def fz_generate_transition(self, opix, npix, time, trans):
    """
        Class-aware wrapper for `::fz_generate_transition()`.
        	Generate a frame of a transition.

        	tpix: Target pixmap
        	opix: Old pixmap
        	npix: New pixmap
        	time: Position within the transition (0 to 256)
        	trans: Transition details

        	Returns 1 if successfully generated a frame.

        	Note: Pixmaps must include alpha.
        """
    return _mupdf.FzPixmap_fz_generate_transition(self, opix, npix, time, trans)