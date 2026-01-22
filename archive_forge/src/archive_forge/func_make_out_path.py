import os
import subprocess
import winreg
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
from itertools import count
def make_out_path(p):
    base, ext = os.path.splitext(p)
    if strip_dir:
        base = os.path.basename(base)
    else:
        _, base = os.path.splitdrive(base)
        if base.startswith((os.path.sep, os.path.altsep)):
            base = base[1:]
    try:
        return os.path.join(output_dir, base + ext_map[ext])
    except LookupError:
        raise CompileError("Don't know how to compile {}".format(p))