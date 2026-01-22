from warnings import warn
from numba.core import types, config, sigutils
from numba.core.errors import DeprecationError, NumbaInvalidConfigWarning
from numba.cuda.compiler import declare_device_function
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.simulator.kernel import FakeCUDAKernel
def autojitwrapper(func):
    return jit(func, device=device, debug=debug, opt=opt, lineinfo=lineinfo, link=link, cache=cache, **kws)