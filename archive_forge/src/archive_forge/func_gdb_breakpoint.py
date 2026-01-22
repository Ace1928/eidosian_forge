import numpy as np
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type
def gdb_breakpoint():
    """
    Calling this function will inject a breakpoint at the call site that is
    recognised by both `gdb` and `gdb_init`, this is to allow breaking at
    multiple points. gdb will stop in the user defined code just after the frame
    employed by the breakpoint returns.
    """
    _gdb_python_call_gen('gdb_breakpoint')()