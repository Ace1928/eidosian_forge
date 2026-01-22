import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_arg_attributes(self):

    def gen_code(attr_name):
        fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(32).as_pointer(), ir.IntType(32)])
        module = ir.Module()
        func = ir.Function(module, fnty, name='sum')
        bb_entry = func.append_basic_block()
        bb_loop = func.append_basic_block()
        bb_exit = func.append_basic_block()
        builder = ir.IRBuilder()
        builder.position_at_end(bb_entry)
        builder.branch(bb_loop)
        builder.position_at_end(bb_loop)
        index = builder.phi(ir.IntType(32))
        index.add_incoming(ir.Constant(index.type, 0), bb_entry)
        accum = builder.phi(ir.IntType(32))
        accum.add_incoming(ir.Constant(accum.type, 0), bb_entry)
        func.args[0].add_attribute(attr_name)
        ptr = builder.gep(func.args[0], [index])
        value = builder.load(ptr)
        added = builder.add(accum, value)
        accum.add_incoming(added, bb_loop)
        indexp1 = builder.add(index, ir.Constant(index.type, 1))
        index.add_incoming(indexp1, bb_loop)
        cond = builder.icmp_unsigned('<', indexp1, func.args[1])
        builder.cbranch(cond, bb_loop, bb_exit)
        builder.position_at_end(bb_exit)
        builder.ret(added)
        return str(module)
    for attr_name in ('byref', 'byval', 'elementtype', 'immarg', 'inalloca', 'inreg', 'nest', 'noalias', 'nocapture', 'nofree', 'nonnull', 'noundef', 'preallocated', 'returned', 'signext', 'swiftasync', 'swifterror', 'swiftself', 'zeroext'):
        llvm.parse_assembly(gen_code(attr_name))