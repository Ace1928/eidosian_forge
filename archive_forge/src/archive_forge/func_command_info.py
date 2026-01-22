import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def command_info(successful_status=(0,), stacklevel=1, **kw):
    info = {}
    for key in kw:
        ok, output = getoutput(kw[key], successful_status=successful_status, stacklevel=stacklevel + 1)
        if ok:
            info[key] = output.strip()
    return info