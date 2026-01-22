import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_initialize_var_map_from_column_order(self):

    class MockConfig(object):
        column_order = None
        file_determinism = FileDeterminism(0)
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([3, 2])
    m.c = Block()
    m.c.x = Var()
    m.c.y = Var([5, 4])
    m.b = Block()
    m.b.x = Var()
    m.b.y = Var([7, 6])
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.column_order = False
    MockConfig.file_determinism = FileDeterminism(0)
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.column_order = True
    MockConfig.file_determinism = FileDeterminism(0)
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]])
    MockConfig.column_order = True
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.column_order = True
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.column_order = True
    MockConfig.file_determinism = FileDeterminism.ORDERED
    var_map = {id(m.b.y[7]): m.b.y[7], id(m.c.y[5]): m.c.y[5], id(m.y[3]): m.y[3]}
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, var_map).values()), [m.b.y[7], m.c.y[5], m.y[3], m.x, m.y[2], m.c.x, m.c.y[4], m.b.x, m.b.y[6]])
    MockConfig.column_order = ComponentMap(((v, i) for i, v in enumerate([m.b.y, m.y, m.c.y[4], m.x])))
    MockConfig.file_determinism = FileDeterminism.ORDERED
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.y[5]])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]])
    MockConfig.column_order = [m.b.y, m.y, m.c.y[4], m.x]
    ref = list(MockConfig.column_order)
    MockConfig.file_determinism = FileDeterminism.ORDERED
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.y[5]])
    self.assertEqual(MockConfig.column_order, ref)
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    self.assertEqual(MockConfig.column_order, ref)
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]])
    self.assertEqual(MockConfig.column_order, ref)