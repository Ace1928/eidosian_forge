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
def fz_new_test_device(is_color, threshold, options, passthrough):
    """
    Class-aware wrapper for `::fz_new_test_device()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_new_test_device(float threshold, int options, ::fz_device *passthrough)` => `(fz_device *, int is_color)`

    	Create a device to test for features.

    	Currently only tests for the presence of non-grayscale colors.

    	is_color: Possible values returned:
    		0: Definitely greyscale
    		1: Probably color (all colors were grey, but there
    		were images or shadings in a non grey colorspace).
    		2: Definitely color

    	threshold: The difference from grayscale that will be tolerated.
    	Typical values to use are either 0 (be exact) and 0.02 (allow an
    	imperceptible amount of slop).

    	options: A set of bitfield options, from the FZ_TEST_OPT set.

    	passthrough: A device to pass all calls through to, or NULL.
    	If set, then the test device can both test and pass through to
    	an underlying device (like, say, the display list device). This
    	means that a display list can be created and at the end we'll
    	know if it's colored or not.

    	In the absence of a passthrough device, the device will throw
    	an exception to stop page interpretation when color is found.
    """
    return _mupdf.fz_new_test_device(is_color, threshold, options, passthrough)