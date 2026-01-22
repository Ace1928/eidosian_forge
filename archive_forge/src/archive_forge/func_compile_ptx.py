from llvmlite import ir
from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, typing, funcdesc, config, compiler, sigutils
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass,
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.typed_passes import (IRLegalization, NativeLowering,
from warnings import warn
from numba.cuda.api import get_current_device
from numba.cuda.target import CUDACABICallConv
@global_compiler_lock
def compile_ptx(pyfunc, sig, debug=False, lineinfo=False, device=False, fastmath=False, cc=None, opt=True, abi='numba', abi_info=None):
    """Compile a Python function to PTX for a given set of argument types.

    :param pyfunc: The Python function to compile.
    :param sig: The signature representing the function's input and output
                types.
    :param debug: Whether to include debug info in the generated PTX.
    :type debug: bool
    :param lineinfo: Whether to include a line mapping from the generated PTX
                     to the source code. Usually this is used with optimized
                     code (since debug mode would automatically include this),
                     so we want debug info in the LLVM but only the line
                     mapping in the final PTX.
    :type lineinfo: bool
    :param device: Whether to compile a device function. Defaults to ``False``,
                   to compile global kernel functions.
    :type device: bool
    :param fastmath: Whether to enable fast math flags (ftz=1, prec_sqrt=0,
                     prec_div=, and fma=1)
    :type fastmath: bool
    :param cc: Compute capability to compile for, as a tuple
               ``(MAJOR, MINOR)``. Defaults to ``(5, 0)``.
    :type cc: tuple
    :param opt: Enable optimizations. Defaults to ``True``.
    :type opt: bool
    :param abi: The ABI for a compiled function - either ``"numba"`` or
                ``"c"``. Note that the Numba ABI is not considered stable.
                The C ABI is only supported for device functions at present.
    :type abi: str
    :param abi_info: A dict of ABI-specific options. The ``"c"`` ABI supports
                     one option, ``"abi_name"``, for providing the wrapper
                     function's name. The ``"numba"`` ABI has no options.
    :type abi_info: dict
    :return: (ptx, resty): The PTX code and inferred return type
    :rtype: tuple
    """
    if abi not in ('numba', 'c'):
        raise NotImplementedError(f'Unsupported ABI: {abi}')
    if abi == 'c' and (not device):
        raise NotImplementedError('The C ABI is not supported for kernels')
    if debug and opt:
        msg = 'debug=True with opt=True (the default) is not supported by CUDA. This may result in a crash - set debug=False or opt=False.'
        warn(NumbaInvalidConfigWarning(msg))
    abi_info = abi_info or dict()
    nvvm_options = {'fastmath': fastmath, 'opt': 3 if opt else 0}
    args, return_type = sigutils.normalize_signature(sig)
    cc = cc or config.CUDA_DEFAULT_PTX_CC
    cres = compile_cuda(pyfunc, return_type, args, debug=debug, lineinfo=lineinfo, fastmath=fastmath, nvvm_options=nvvm_options, cc=cc)
    resty = cres.signature.return_type
    if resty and (not device) and (resty != types.void):
        raise TypeError('CUDA kernel must have void return type.')
    tgt = cres.target_context
    if device:
        lib = cres.library
        if abi == 'c':
            wrapper_name = abi_info.get('abi_name', pyfunc.__name__)
            lib = cabi_wrap_function(tgt, lib, cres.fndesc, wrapper_name, nvvm_options)
    else:
        code = pyfunc.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno
        lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug, lineinfo, nvvm_options, filename, linenum)
    ptx = lib.get_asm_str(cc=cc)
    return (ptx, resty)