import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def create_target_machine(self, cpu='', features='', opt=2, reloc='default', codemodel='jitdefault', printmc=False, jit=False, abiname=''):
    """
        Create a new TargetMachine for this target and the given options.

        Specifying codemodel='default' will result in the use of the "small"
        code model. Specifying codemodel='jitdefault' will result in the code
        model being picked based on platform bitness (32="small", 64="large").

        The `printmc` option corresponds to llvm's `-print-machineinstrs`.

        The `jit` option should be set when the target-machine is to be used
        in a JIT engine.

        The `abiname` option specifies the ABI. RISC-V targets with hard-float
        needs to pass the ABI name to LLVM.
        """
    assert 0 <= opt <= 3
    assert reloc in RELOC
    assert codemodel in CODEMODEL
    triple = self._triple
    if os.name == 'nt' and codemodel == 'jitdefault':
        triple += '-elf'
    tm = ffi.lib.LLVMPY_CreateTargetMachine(self, _encode_string(triple), _encode_string(cpu), _encode_string(features), opt, _encode_string(reloc), _encode_string(codemodel), int(printmc), int(jit), _encode_string(abiname))
    if tm:
        return TargetMachine(tm)
    else:
        raise RuntimeError('Cannot create target machine')