import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def inspect_cfg(self, signature=None, show_wrapper=None, **kwargs):
    """
        For inspecting the CFG of the function.

        By default the CFG of the user function is shown.  The *show_wrapper*
        option can be set to "python" or "cfunc" to show the python wrapper
        function or the *cfunc* wrapper function, respectively.

        Parameters accepted in kwargs
        -----------------------------
        filename : string, optional
            the name of the output file, if given this will write the output to
            filename
        view : bool, optional
            whether to immediately view the optional output file
        highlight : bool, set, dict, optional
            what, if anything, to highlight, options are:
            { incref : bool, # highlight NRT_incref calls
              decref : bool, # highlight NRT_decref calls
              returns : bool, # highlight exits which are normal returns
              raises : bool, # highlight exits which are from raise
              meminfo : bool, # highlight calls to NRT*meminfo
              branches : bool, # highlight true/false branches
             }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'incref', 'decref'} would
            switch on highlighting on increfs and decrefs.
        interleave: bool, set, dict, optional
            what, if anything, to interleave in the LLVM IR, options are:
            { python: bool # interleave python source code with the LLVM IR
              lineinfo: bool # interleave line information markers with the LLVM
                             # IR
            }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'python',} would
            switch on interleaving of python source code in the LLVM IR.
        strip_ir : bool, optional
            Default is False. If set to True all LLVM IR that is superfluous to
            that requested in kwarg `highlight` will be removed.
        show_key : bool, optional
            Default is True. Create a "key" for the highlighting in the rendered
            CFG.
        fontsize : int, optional
            Default is 8. Set the fontsize in the output to this value.
        """
    if signature is not None:
        cres = self.overloads[signature]
        lib = cres.library
        if show_wrapper == 'python':
            fname = cres.fndesc.llvm_cpython_wrapper_name
        elif show_wrapper == 'cfunc':
            fname = cres.fndesc.llvm_cfunc_wrapper_name
        else:
            fname = cres.fndesc.mangled_name
        return lib.get_function_cfg(fname, py_func=self.py_func, **kwargs)
    return dict(((sig, self.inspect_cfg(sig, show_wrapper=show_wrapper)) for sig in self.signatures))