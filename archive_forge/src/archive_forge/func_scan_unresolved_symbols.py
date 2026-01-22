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
def scan_unresolved_symbols(self, module, engine):
    """
        Scan and track all unresolved external symbols in the module and
        allocate memory for it.
        """
    prefix = self.PREFIX
    for gv in module.global_variables:
        if gv.name.startswith(prefix):
            sym = gv.name[len(prefix):]
            if engine.is_symbol_defined(gv.name):
                continue
            abortfn = rtsys.library.get_pointer_to_function('nrt_unresolved_abort')
            ptr = ctypes.c_void_p(abortfn)
            engine.add_global_mapping(gv, ctypes.addressof(ptr))
            self._unresolved[sym] = ptr