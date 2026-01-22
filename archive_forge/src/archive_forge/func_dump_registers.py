import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def dump_registers(cls, registers, arch=None):
    """
        Dump the x86/x64 processor register values.
        The output mimics that of the WinDBG debugger.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.
            Currently only the following architectures are supported:
             - L{win32.ARCH_I386}
             - L{win32.ARCH_AMD64}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if registers is None:
        return ''
    if arch is None:
        if 'Eax' in registers:
            arch = win32.ARCH_I386
        elif 'Rax' in registers:
            arch = win32.ARCH_AMD64
        else:
            arch = 'Unknown'
    if arch not in cls.reg_template:
        msg = "Don't know how to dump the registers for architecture: %s"
        raise NotImplementedError(msg % arch)
    registers = registers.copy()
    registers['efl_dump'] = cls.dump_flags(registers['EFlags'])
    return cls.reg_template[arch] % registers