import tempfile
import os
import pickle
import random
import collections
import itertools
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import (
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.suffix import suffix
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.parameter import parameter, parameter_dict, parameter_list
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.variable import IVariable, variable, variable_dict, variable_list
from pyomo.core.kernel.block import IBlock, block, block_dict, block_tuple, block_list
from pyomo.core.kernel.sos import sos
from pyomo.opt.results import Solution
def test_clone1(self):
    b = block()
    b.v = variable()
    b.b = block()
    b.b.e = expression(b.v ** 2)
    b.b.v = variable()
    b.bdict = block_dict()
    b.bdict[0] = block()
    b.bdict[0].e = expression(b.v ** 2)
    b.blist = block_list()
    b.blist.append(block())
    b.blist[0].e = expression(b.v ** 2 + b.b.v ** 2)
    bc = b.clone()
    self.assertIsNot(b.v, bc.v)
    self.assertIsNot(b.b.e, bc.b.e)
    self.assertIsNot(b.bdict[0].e, bc.bdict[0].e)
    self.assertIsNot(b.blist[0].e, bc.blist[0].e)
    b_b = b.b.clone()
    self.assertIsNot(b_b.e, b.b.e)
    self.assertTrue(len(_collect_expr_components(b.b.e.expr)) == 1)
    self.assertIs(list(_collect_expr_components(b.b.e.expr).values())[0], b.v)
    self.assertTrue(len(_collect_expr_components(b_b.e.expr)) == 1)
    self.assertIs(list(_collect_expr_components(b_b.e.expr).values())[0], b.v)
    b_bdict0 = b.bdict[0].clone()
    self.assertIsNot(b_bdict0.e, b.bdict[0].e)
    self.assertTrue(len(_collect_expr_components(b.bdict[0].e.expr)) == 1)
    self.assertIs(list(_collect_expr_components(b.bdict[0].e.expr).values())[0], b.v)
    self.assertTrue(len(_collect_expr_components(b_bdict0.e.expr)) == 1)
    self.assertIs(list(_collect_expr_components(b_bdict0.e.expr).values())[0], b.v)
    b_blist0 = b.blist[0].clone()
    self.assertIsNot(b_blist0.e, b.blist[0].e)
    self.assertTrue(len(_collect_expr_components(b.blist[0].e.expr)) == 2)
    self.assertEqual(sorted(list((id(v_) for v_ in _collect_expr_components(b.blist[0].e.expr).values()))), sorted(list((id(v_) for v_ in [b.v, b.b.v]))))
    self.assertTrue(len(_collect_expr_components(b_blist0.e.expr)) == 2)
    self.assertEqual(sorted(list((id(v_) for v_ in _collect_expr_components(b_blist0.e.expr).values()))), sorted(list((id(v_) for v_ in [b.v, b.b.v]))))