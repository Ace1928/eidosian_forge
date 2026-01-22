import sys
import os
from io import StringIO

    Acquire a readable object for a given package name and identifier.
    An IOError will be raised if the resource cannot be found.

    For example::

        mydata = get_resource('mypkgdata.jpg').read()

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.

    In some cases, getResource will return a real file object.  In that
    case, it may be useful to use its name attribute to get the path
    rather than use it as a file-like object.  For example, you may
    be handing data off to a C API.
    