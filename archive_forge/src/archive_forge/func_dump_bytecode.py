import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
def dump_bytecode(code, lineno=False):
    """
    Use this function to write unit tests: copy/paste its output to
    write a self.assertBlocksEqual() check.
    """
    print()
    if isinstance(code, (Bytecode, ConcreteBytecode)):
        is_concrete = isinstance(code, ConcreteBytecode)
        if is_concrete:
            block = list(code)
        else:
            block = code
        indent = ' ' * 8
        labels = {}
        for index, instr in enumerate(block):
            if isinstance(instr, Label):
                name = 'label_instr%s' % index
                labels[instr] = name
        if is_concrete:
            name = 'ConcreteBytecode'
            print(indent + 'code = %s()' % name)
            if code.argcount:
                print(indent + 'code.argcount = %s' % code.argcount)
            if sys.version_info > (3, 8):
                if code.posonlyargcount:
                    print(indent + 'code.posonlyargcount = %s' % code.posonlyargcount)
            if code.kwonlyargcount:
                print(indent + 'code.kwargonlycount = %s' % code.kwonlyargcount)
            print(indent + 'code.flags = %#x' % code.flags)
            if code.consts:
                print(indent + 'code.consts = %r' % code.consts)
            if code.names:
                print(indent + 'code.names = %r' % code.names)
            if code.varnames:
                print(indent + 'code.varnames = %r' % code.varnames)
        for name in sorted(labels.values()):
            print(indent + '%s = Label()' % name)
        if is_concrete:
            text = indent + 'code.extend('
            indent = ' ' * len(text)
        else:
            text = indent + 'code = Bytecode('
            indent = ' ' * len(text)
        lines = _format_instr_list(code, labels, lineno).splitlines()
        last_line = len(lines) - 1
        for index, line in enumerate(lines):
            if index == 0:
                print(text + lines[0])
            elif index == last_line:
                print(indent + line + ')')
            else:
                print(indent + line)
        print()
    else:
        assert isinstance(code, ControlFlowGraph)
        labels = {}
        for block_index, block in enumerate(code):
            labels[id(block)] = 'code[%s]' % block_index
        for block_index, block in enumerate(code):
            text = _format_instr_list(block, labels, lineno)
            if block_index != len(code) - 1:
                text += ','
            print(text)
            print()