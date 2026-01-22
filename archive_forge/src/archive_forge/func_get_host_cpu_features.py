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
def get_host_cpu_features():
    """Get host CPU features using LLVM.

    The features may be modified due to user setting.
    See numba.config.ENABLE_AVX.
    """
    try:
        features = ll.get_host_cpu_features()
    except RuntimeError:
        return ''
    else:
        if not config.ENABLE_AVX:
            for k in features:
                if k.startswith('avx'):
                    features[k] = False
        return features.flatten()