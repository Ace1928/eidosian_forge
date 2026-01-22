from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def get_cubin(self, cc=None):
    if cc is None:
        ctx = devices.get_context()
        device = ctx.device
        cc = device.compute_capability
    cubin = self._cubin_cache.get(cc, None)
    if cubin:
        return cubin
    linker = driver.Linker.new(max_registers=self._max_registers, cc=cc)
    ptxes = self._get_ptxes(cc=cc)
    for ptx in ptxes:
        linker.add_ptx(ptx.encode())
    for path in self._linking_files:
        linker.add_file_guess_ext(path)
    if self.needs_cudadevrt:
        linker.add_file_guess_ext(get_cudalib('cudadevrt', static=True))
    cubin = linker.complete()
    self._cubin_cache[cc] = cubin
    self._linkerinfo_cache[cc] = linker.info_log
    return cubin