from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
def fullReport(self, bShowNotes=True):
    """
        @type  bShowNotes: bool
        @param bShowNotes: C{True} to show the user notes, C{False} otherwise.

        @rtype:  str
        @return: Long description of the event.
        """
    msg = self.briefReport()
    msg += '\n'
    if self.bits == 32:
        width = 16
    else:
        width = 8
    if self.eventCode == win32.EXCEPTION_DEBUG_EVENT:
        exploitability, expcode, expdescription = self.isExploitable()
        msg += '\nSecurity risk level: %s\n' % exploitability
        msg += '  %s\n' % expdescription
    if bShowNotes and self.notes:
        msg += '\nNotes:\n'
        msg += self.notesReport()
    if self.commandLine:
        msg += '\nCommand line: %s\n' % self.commandLine
    if self.environment:
        msg += '\nEnvironment:\n'
        msg += self.environmentReport()
    if not self.labelPC:
        base = HexDump.address(self.lpBaseOfDll, self.bits)
        if self.modFileName:
            fn = PathOperations.pathname_to_filename(self.modFileName)
            msg += '\nRunning in %s (%s)\n' % (fn, base)
        else:
            msg += '\nRunning in module at %s\n' % base
    if self.registers:
        msg += '\nRegisters:\n'
        msg += CrashDump.dump_registers(self.registers)
        if self.registersPeek:
            msg += '\n'
            msg += CrashDump.dump_registers_peek(self.registers, self.registersPeek, width=width)
    if self.faultDisasm:
        msg += '\nCode disassembly:\n'
        msg += CrashDump.dump_code(self.faultDisasm, self.pc, bits=self.bits)
    if self.stackTrace:
        msg += '\nStack trace:\n'
        if self.stackTracePretty:
            msg += CrashDump.dump_stack_trace_with_labels(self.stackTracePretty, bits=self.bits)
        else:
            msg += CrashDump.dump_stack_trace(self.stackTrace, bits=self.bits)
    if self.stackFrame:
        if self.stackPeek:
            msg += '\nStack pointers:\n'
            msg += CrashDump.dump_stack_peek(self.stackPeek, width=width)
        msg += '\nStack dump:\n'
        msg += HexDump.hexblock(self.stackFrame, self.sp, bits=self.bits, width=width)
    if self.faultCode and (not self.modFileName):
        msg += '\nCode dump:\n'
        msg += HexDump.hexblock(self.faultCode, self.pc, bits=self.bits, width=width)
    if self.faultMem:
        if self.faultPeek:
            msg += '\nException address pointers:\n'
            msg += CrashDump.dump_data_peek(self.faultPeek, self.exceptionAddress, bits=self.bits, width=width)
        msg += '\nException address dump:\n'
        msg += HexDump.hexblock(self.faultMem, self.exceptionAddress, bits=self.bits, width=width)
    if self.memoryMap:
        msg += '\nMemory map:\n'
        mappedFileNames = dict()
        for mbi in self.memoryMap:
            if hasattr(mbi, 'filename') and mbi.filename:
                mappedFileNames[mbi.BaseAddress] = mbi.filename
        msg += CrashDump.dump_memory_map(self.memoryMap, mappedFileNames, bits=self.bits)
    if not msg.endswith('\n\n'):
        if not msg.endswith('\n'):
            msg += '\n'
        msg += '\n'
    return msg