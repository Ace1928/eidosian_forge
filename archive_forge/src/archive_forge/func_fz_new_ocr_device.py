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
def fz_new_ocr_device(self, ctm, mediabox, with_list, language, datadir, progress, progress_arg):
    """
        Class-aware wrapper for `::fz_new_ocr_device()`.
        	Create a device to OCR the text on the page.

        	Renders the page internally to a bitmap that is then OCRd. Text
        	is then forwarded onto the target device.

        	target: The target device to receive the OCRd text.

        	ctm: The transform to apply to the mediabox to get the size for
        	the rendered page image. Also used to calculate the resolution
        	for the page image. In general, this will be the same as the CTM
        	that you pass to fz_run_page (or fz_run_display_list) to feed
        	this device.

        	mediabox: The mediabox (in points). Combined with the CTM to get
        	the bounds of the pixmap used internally for the rendered page
        	image.

        	with_list: If with_list is false, then all non-text operations
        	are forwarded instantly to the target device. This results in
        	the target device seeing all NON-text operations, followed by
        	all the text operations (derived from OCR).

        	If with_list is true, then all the marking operations are
        	collated into a display list which is then replayed to the
        	target device at the end.

        	language: NULL (for "eng"), or a pointer to a string to describe
        	the languages/scripts that should be used for OCR (e.g.
        	"eng,ara").

        	datadir: NULL (for ""), or a pointer to a path string otherwise
        	provided to Tesseract in the TESSDATA_PREFIX environment variable.

        	progress: NULL, or function to be called periodically to indicate
        	progress. Return 0 to continue, or 1 to cancel. progress_arg is
        	returned as the void *. The int is a value between 0 and 100 to
        	indicate progress.

        	progress_arg: A void * value to be parrotted back to the progress
        	function.
        """
    return _mupdf.FzDevice_fz_new_ocr_device(self, ctm, mediabox, with_list, language, datadir, progress, progress_arg)