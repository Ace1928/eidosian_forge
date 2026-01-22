import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic

    Adds the Numba break point into the source
    