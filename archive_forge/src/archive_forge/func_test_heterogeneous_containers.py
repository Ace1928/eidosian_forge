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
def test_heterogeneous_containers(self):
    order = list((str(obj) for obj in heterogeneous_containers(self.model.V)))
    self.assertEqual(order, [])
    order = list((str(obj) for obj in heterogeneous_containers(self.model.v)))
    self.assertEqual(order, [])
    order = list((str(obj) for obj in heterogeneous_containers(self.model)))
    self.assertEqual(order, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'K[0].b', 'K[0].B[0]', 'K[0].B[1][0]', 'K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'K[0].J[1][0].b', 'j', 'J[0]', 'J[1][0]', 'J[1][0].b'])

    def f(x):
        self.assertTrue(x._is_heterogeneous_container)
        parent = x.parent
        while parent is not None:
            if parent is self.model:
                return False
            parent = parent.parent
        return True
    order1 = list((str(obj) for obj in heterogeneous_containers(self.model, descend_into=f)))
    order2 = list((str(obj) for obj in heterogeneous_containers(self.model, descend_into=lambda x: True if x is self.model else False)))
    self.assertEqual(order1, order2)
    self.assertEqual(order1, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'j', 'J[0]', 'J[1][0]'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model, ctype=IBlock)))
    self.assertEqual(order, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'K[0].b', 'K[0].B[0]', 'K[0].B[1][0]', 'K[0].J[1][0].b', 'J[1][0].b'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model, ctype=IJunk)))
    self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'j', 'J[0]', 'J[1][0]'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model.K, ctype=IJunk)))
    self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0], ctype=IJunk)))
    self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0].j, ctype=IJunk)))
    self.assertEqual(order, ['K[0].j'])
    order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0].j, ctype=IBlock)))
    self.assertEqual(order, [])