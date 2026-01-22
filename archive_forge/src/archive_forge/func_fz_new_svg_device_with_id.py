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
def fz_new_svg_device_with_id(self, page_width, page_height, text_format, reuse_images, id):
    """
        Class-aware wrapper for `::fz_new_svg_device_with_id()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_new_svg_device_with_id(float page_width, float page_height, int text_format, int reuse_images)` => `(fz_device *, int id)`

        	Create a device that outputs (single page) SVG files to
        	the given output stream.

        	output: The output stream to send the constructed SVG page to.

        	page_width, page_height: The page dimensions to use (in points).

        	text_format: How to emit text. One of the following values:
        		FZ_SVG_TEXT_AS_TEXT: As <text> elements with possible
        		layout errors and mismatching fonts.
        		FZ_SVG_TEXT_AS_PATH: As <path> elements with exact
        		visual appearance.

        	reuse_images: Share image resources using <symbol> definitions.

        	id: ID parameter to keep generated IDs unique across SVG files.
        """
    return _mupdf.FzOutput_fz_new_svg_device_with_id(self, page_width, page_height, text_format, reuse_images, id)