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
def ll_fz_parse_page_range(s, n):
    ret = ll_fz_parse_page_range_orig(s, n)
    if len(ret) == 2:
        return (None, 0, 0)
    else:
        return (ret[0], ret[1], ret[2])