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
def fz_image_orientation(self):
    """
        Class-aware wrapper for `::fz_image_orientation()`.
        	Request the natural orientation of an image.

        	This is for images (such as JPEG) that can contain internal
        	specifications of rotation/flips. This is ignored by all the
        	internal decode/rendering routines, but can be used by callers
        	(such as the image document handler) to respect such
        	specifications.

        	The values used by MuPDF are as follows, with the equivalent
        	Exif specifications given for information:

        	0: Undefined
        	1: 0 degree ccw rotation. (Exif = 1)
        	2: 90 degree ccw rotation. (Exif = 8)
        	3: 180 degree ccw rotation. (Exif = 3)
        	4: 270 degree ccw rotation. (Exif = 6)
        	5: flip on X. (Exif = 2)
        	6: flip on X, then rotate ccw by 90 degrees. (Exif = 5)
        	7: flip on X, then rotate ccw by 180 degrees. (Exif = 4)
        	8: flip on X, then rotate ccw by 270 degrees. (Exif = 7)
        """
    return _mupdf.FzImage_fz_image_orientation(self)