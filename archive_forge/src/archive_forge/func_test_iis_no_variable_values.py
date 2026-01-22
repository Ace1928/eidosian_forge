import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import (
def test_iis_no_variable_values(self):
    with ReaderFactory('sol') as reader:
        if reader is None:
            raise IOError("Reader 'sol' is not registered")
        result = reader(join(currdir, 'iis_no_variable_values.sol'), suffixes=['iis'])
        soln = result.solution(0)
        self.assertEqual(len(list(soln.variable['v0'].keys())), 1)
        self.assertEqual(soln.variable['v0']['iis'], 1)
        self.assertEqual(len(list(soln.variable['v1'].keys())), 1)
        self.assertEqual(soln.variable['v1']['iis'], 1)
        self.assertEqual(len(list(soln.constraint['c0'].keys())), 1)
        self.assertEqual(soln.constraint['c0']['Iis'], 4)
        import pyomo.kernel as pmo
        m = pmo.block()
        m.v0 = pmo.variable()
        m.v1 = pmo.variable()
        m.c0 = pmo.constraint()
        m.iis = pmo.suffix(direction=pmo.suffix.IMPORT)
        from pyomo.core.expr.symbol_map import SymbolMap
        soln.symbol_map = SymbolMap()
        soln.symbol_map.addSymbol(m.v0, 'v0')
        soln.symbol_map.addSymbol(m.v1, 'v1')
        soln.symbol_map.addSymbol(m.c0, 'c0')
        m.load_solution(soln)
        pmo.pprint(m.iis)
        self.assertEqual(m.iis[m.v0], 1)
        self.assertEqual(m.iis[m.v1], 1)
        self.assertEqual(m.iis[m.c0], 4)