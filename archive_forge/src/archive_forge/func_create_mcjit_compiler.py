import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def create_mcjit_compiler(module, target_machine, use_lmm=None):
    """
    Create a MCJIT ExecutionEngine from the given *module* and
    *target_machine*.

    *lmm* controls whether the llvmlite memory manager is used. If not supplied,
    the default choice for the platform will be used (``True`` on 64-bit ARM
    systems, ``False`` otherwise).
    """
    if use_lmm is None:
        use_lmm = platform.machine() in ('arm64', 'aarch64')
    with ffi.OutputString() as outerr:
        engine = ffi.lib.LLVMPY_CreateMCJITCompiler(module, target_machine, use_lmm, outerr)
        if not engine:
            raise RuntimeError(str(outerr))
    target_machine._owned = True
    return ExecutionEngine(engine, module=module)