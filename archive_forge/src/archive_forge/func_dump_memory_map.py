import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_memory_map(memoryMap, mappedFilenames=None, bits=None):
    """
        Dump the memory map of a process. Optionally show the filenames for
        memory mapped files as well.

        @type  memoryMap: list( L{win32.MemoryBasicInformation} )
        @param memoryMap: Memory map returned by L{Process.get_memory_map}.

        @type  mappedFilenames: dict( int S{->} str )
        @param mappedFilenames: (Optional) Memory mapped filenames
            returned by L{Process.get_mapped_filenames}.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if not memoryMap:
        return ''
    table = Table()
    if mappedFilenames:
        table.addRow('Address', 'Size', 'State', 'Access', 'Type', 'File')
    else:
        table.addRow('Address', 'Size', 'State', 'Access', 'Type')
    for mbi in memoryMap:
        BaseAddress = HexDump.address(mbi.BaseAddress, bits)
        RegionSize = HexDump.address(mbi.RegionSize, bits)
        mbiState = mbi.State
        if mbiState == win32.MEM_RESERVE:
            State = 'Reserved'
        elif mbiState == win32.MEM_COMMIT:
            State = 'Commited'
        elif mbiState == win32.MEM_FREE:
            State = 'Free'
        else:
            State = 'Unknown'
        if mbiState != win32.MEM_COMMIT:
            Protect = ''
        else:
            mbiProtect = mbi.Protect
            if mbiProtect & win32.PAGE_NOACCESS:
                Protect = '--- '
            elif mbiProtect & win32.PAGE_READONLY:
                Protect = 'R-- '
            elif mbiProtect & win32.PAGE_READWRITE:
                Protect = 'RW- '
            elif mbiProtect & win32.PAGE_WRITECOPY:
                Protect = 'RC- '
            elif mbiProtect & win32.PAGE_EXECUTE:
                Protect = '--X '
            elif mbiProtect & win32.PAGE_EXECUTE_READ:
                Protect = 'R-X '
            elif mbiProtect & win32.PAGE_EXECUTE_READWRITE:
                Protect = 'RWX '
            elif mbiProtect & win32.PAGE_EXECUTE_WRITECOPY:
                Protect = 'RCX '
            else:
                Protect = '??? '
            if mbiProtect & win32.PAGE_GUARD:
                Protect += 'G'
            else:
                Protect += '-'
            if mbiProtect & win32.PAGE_NOCACHE:
                Protect += 'N'
            else:
                Protect += '-'
            if mbiProtect & win32.PAGE_WRITECOMBINE:
                Protect += 'W'
            else:
                Protect += '-'
        mbiType = mbi.Type
        if mbiType == win32.MEM_IMAGE:
            Type = 'Image'
        elif mbiType == win32.MEM_MAPPED:
            Type = 'Mapped'
        elif mbiType == win32.MEM_PRIVATE:
            Type = 'Private'
        elif mbiType == 0:
            Type = ''
        else:
            Type = 'Unknown'
        if mappedFilenames:
            FileName = mappedFilenames.get(mbi.BaseAddress, '')
            table.addRow(BaseAddress, RegionSize, State, Protect, Type, FileName)
        else:
            table.addRow(BaseAddress, RegionSize, State, Protect, Type)
    return table.getOutput()