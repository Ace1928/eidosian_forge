from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def test_DILocation(self):
    """ Tests that DILocation information is reasonable.
        """

    @njit(debug=True, error_model='numpy')
    def foo(a):
        b = a + 1.23
        c = b * 2.34
        d = b / c
        print(d)
        return d
    sig = (types.float64,)
    metadata = self._get_metadata(foo, sig=sig)
    full_ir = self._get_llvmir(foo, sig=sig)
    module = llvm.parse_assembly(full_ir)
    name = foo.overloads[foo.signatures[0]].fndesc.mangled_name
    funcs = [x for x in module.functions if x.name == name]
    self.assertEqual(len(funcs), 1)
    func = funcs[0]
    blocks = [x for x in func.blocks]
    self.assertGreater(len(blocks), 1)
    block = blocks[0]
    instrs = [x for x in block.instructions if x.opcode != 'call']
    op_expect = {'fadd', 'fmul', 'fdiv'}
    started = False
    for x in instrs:
        if x.opcode in op_expect:
            op_expect.remove(x.opcode)
            if not started:
                started = True
        elif op_expect and started:
            self.fail('Math opcodes are not contiguous')
    self.assertFalse(op_expect, 'Math opcodes were not found')
    line2dbg = set()
    re_dbg_ref = re.compile('.*!dbg (![0-9]+).*$')
    found = -1
    for instr in instrs:
        inst_as_str = str(instr)
        matched = re_dbg_ref.match(inst_as_str)
        if not matched:
            accepted = ('alloca ', 'store ')
            self.assertTrue(any([x in inst_as_str for x in accepted]))
            continue
        groups = matched.groups()
        self.assertEqual(len(groups), 1)
        dbg_val = groups[0]
        int_dbg_val = int(dbg_val[1:])
        if found >= 0:
            self.assertTrue(int_dbg_val >= found)
        found = int_dbg_val
        line2dbg.add(dbg_val)
    pysrc, pysrc_line_start = inspect.getsourcelines(foo)
    metadata_definition_map = self._get_metadata_map(metadata)
    offsets = [0, 1, 2, 3]
    pyln_range = [pysrc_line_start + 2 + x for x in offsets]
    for k, line_no in zip(sorted(line2dbg, key=lambda x: int(x[1:])), pyln_range):
        dilocation_info = metadata_definition_map[k]
        self.assertIn(f'line: {line_no}', dilocation_info)
    expr = '.*!DILocalVariable\\(name: "a",.*line: ([0-9]+),.*'
    match_local_var_a = re.compile(expr)
    for entry in metadata_definition_map.values():
        matched = match_local_var_a.match(entry)
        if matched:
            groups = matched.groups()
            self.assertEqual(len(groups), 1)
            dbg_line = int(groups[0])
            defline = pysrc_line_start + 1
            self.assertEqual(dbg_line, defline)
            break
    else:
        self.fail('Assertion on DILocalVariable not made')