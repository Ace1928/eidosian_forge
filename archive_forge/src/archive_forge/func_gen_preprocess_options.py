import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def gen_preprocess_options(macros, include_dirs):
    """Generate C pre-processor options (-D, -U, -I) as used by at least
    two types of compilers: the typical Unix compiler and Visual C++.
    'macros' is the usual thing, a list of 1- or 2-tuples, where (name,)
    means undefine (-U) macro 'name', and (name,value) means define (-D)
    macro 'name' to 'value'.  'include_dirs' is just a list of directory
    names to be added to the header file search path (-I).  Returns a list
    of command-line options suitable for either Unix compilers or Visual
    C++.
    """
    pp_opts = []
    for macro in macros:
        if not (isinstance(macro, tuple) and 1 <= len(macro) <= 2):
            raise TypeError("bad macro definition '%s': each element of 'macros' list must be a 1- or 2-tuple" % macro)
        if len(macro) == 1:
            pp_opts.append('-U%s' % macro[0])
        elif len(macro) == 2:
            if macro[1] is None:
                pp_opts.append('-D%s' % macro[0])
            else:
                pp_opts.append('-D%s=%s' % macro)
    for dir in include_dirs:
        pp_opts.append('-I%s' % dir)
    return pp_opts