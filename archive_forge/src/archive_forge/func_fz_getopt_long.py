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
def fz_getopt_long(nargc, nargv, ostr, longopts):
    """
    Class-aware wrapper for `::fz_getopt_long()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_getopt_long(int nargc, const char *ostr, const ::fz_getopt_long_options *longopts)` => `(int, char *nargv)`

    	Simple functions/variables for use in tools.

    	ostr = option string. Comprises single letter options, followed by : if there
    	is an argument to the option.

    	longopts: NULL (indicating no long options), or a pointer to an array of
    	longoptions, terminated by an entry with option == NULL.

    	In the event of matching a single char option, this function will normally
    	return the char. The exception to this is when the option requires an
    	argument and none is supplied; in this case we return ':'.

    	In the event of matching a long option, this function returns 0, with fz_optlong
    	set to point to the matching option.

    	A long option entry may be followed with : to indicate there is an argument to the
    	option. If the need for an argument is specified in this way, and no argument is
    	given, an error will be displayed and argument processing will stop. If an argument
    	is given, and the long option record contains a non-null flag pointer, then the code
    	will decode the argument and fill in that flag pointer. Specifically,
    	case-insensitive matches to 'yes', 'no', 'true' and 'false' will cause a value of 0
    	or 1 as appropriate to be written; failing this the arg will be interpreted as a
    	decimal integer using atoi.

    	A long option entry may be followed by an list of options (e.g. myoption=foo|bar|baz)
    	and the option will be passed to fz_opt_from_list. The return value of that will be
    	placed in fz_optitem. If the return value of that function is -1, then an error will
    	be displayed and argument processing will stop.

    	In the event of reaching the end of the arg list or '--', this function returns EOF.

    	In the event of failing to match anything, an error is printed, and we return '?'.

    	If an argument is expected for the option, then fz_optarg will be returned pointing
    	at the start of the argument. Examples of supported argument formats: '-r500', '-r 500',
    	'--resolution 500', '--resolution=500'.
    """
    return _mupdf.fz_getopt_long(nargc, nargv, ostr, longopts)