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
def fz_new_draw_device_with_bbox_proof(transform, dest, clip, cs):
    """
    Class-aware wrapper for `::fz_new_draw_device_with_bbox_proof()`.
    	Create a device to draw on a pixmap.

    	dest: Target pixmap for the draw device. See fz_new_pixmap*
    	for how to obtain a pixmap. The pixmap is not cleared by the
    	draw device, see fz_clear_pixmap* for how to clear it prior to
    	calling fz_new_draw_device. Free the device by calling
    	fz_drop_device.

    	transform: Transform from user space in points to device space
    	in pixels.

    	clip: Bounding box to restrict any marking operations of the
    	draw device.

    	proof_cs: Color space to render to prior to mapping to color
    	space defined by pixmap.
    """
    return _mupdf.fz_new_draw_device_with_bbox_proof(transform, dest, clip, cs)