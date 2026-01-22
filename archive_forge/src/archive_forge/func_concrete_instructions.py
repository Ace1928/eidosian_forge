import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
def concrete_instructions(self):
    ncells = len(self.bytecode.cellvars)
    lineno = self.bytecode.first_lineno
    for instr in self.bytecode:
        if isinstance(instr, Label):
            self.labels[instr] = len(self.instructions)
            continue
        if isinstance(instr, SetLineno):
            lineno = instr.lineno
            continue
        if isinstance(instr, ConcreteInstr):
            instr = instr.copy()
        else:
            assert isinstance(instr, Instr)
            if instr.lineno is not None:
                lineno = instr.lineno
            arg = instr.arg
            is_jump = isinstance(arg, Label)
            if is_jump:
                label = arg
                arg = 0
            elif instr.opcode in _opcode.hasconst:
                arg = self.add_const(arg)
            elif instr.opcode in _opcode.haslocal:
                arg = self.add(self.varnames, arg)
            elif instr.opcode in _opcode.hasname:
                arg = self.add(self.names, arg)
            elif instr.opcode in _opcode.hasfree:
                if isinstance(arg, CellVar):
                    arg = self.bytecode.cellvars.index(arg.name)
                else:
                    assert isinstance(arg, FreeVar)
                    arg = ncells + self.bytecode.freevars.index(arg.name)
            elif instr.opcode in _opcode.hascompare:
                if isinstance(arg, Compare):
                    arg = arg.value
            instr = ConcreteInstr(instr.name, arg, lineno=lineno)
            if is_jump:
                self.jumps.append((len(self.instructions), label, instr))
        self.instructions.append(instr)