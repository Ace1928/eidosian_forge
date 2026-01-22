import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_loop_dfs_pathological(self):
    g = self.from_adj_list({0: {38, 14}, 14: {38, 22}, 22: {38, 30}, 30: {42, 38}, 38: {42}, 42: {64, 50}, 50: {64, 58}, 58: {128}, 64: {72, 86}, 72: {80, 86}, 80: {128}, 86: {108, 94}, 94: {108, 102}, 102: {128}, 108: {128, 116}, 116: {128, 124}, 124: {128}, 128: {178, 174}, 174: {178}, 178: {210, 206}, 206: {210}, 210: {248, 252}, 248: {252}, 252: {282, 286}, 282: {286}, 286: {296, 326}, 296: {330}, 326: {330}, 330: {370, 340}, 340: {374}, 370: {374}, 374: {380, 382}, 380: {382}, 382: {818, 390}, 390: {456, 458}, 456: {458}, 458: {538, 566}, 538: {548, 566}, 548: set(), 566: {586, 572}, 572: {586}, 586: {708, 596}, 596: {608}, 608: {610}, 610: {704, 620}, 620: {666, 630}, 630: {636, 646}, 636: {666, 646}, 646: {666}, 666: {610}, 704: {706}, 706: {818}, 708: {720}, 720: {722}, 722: {816, 732}, 732: {778, 742}, 742: {748, 758}, 748: {778, 758}, 758: {778}, 778: {722}, 816: {818}, 818: set()})
    g.set_entry_point(0)
    g.process()
    stats = {}
    back_edges = g._find_back_edges(stats=stats)
    self.assertEqual(back_edges, {(666, 610), (778, 722)})
    self.assertEqual(stats['iteration_count'], 155)