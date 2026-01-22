import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
def get_pointer_to_function(self, name):
    """
        Generate native code for function named *name* and return a pointer
        to the start of the function (as an integer).

        This function implicitly calls .finalize().

        Returns
        -------
        pointer : int
            - zero (null) if no symbol of *name* is defined by this code
              library.
            - non-zero if the symbol is defined.
        """
    self._ensure_finalized()
    ee = self._codegen._engine
    if not ee.is_symbol_defined(name):
        return 0
    else:
        return self._codegen._engine.get_function_address(name)