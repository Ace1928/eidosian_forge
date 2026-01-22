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
def test_components_active_True(self):
    for obj in self._components:
        self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
        self.assertTrue(isinstance(obj, IBlock))
        self.assertEqual(sorted((str(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=True))), sorted((str(_b) for _b in self._components[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else [])
        self.assertEqual(set((id(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=True))), set((id(_b) for _b in self._components[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else set())
        self.assertEqual(sorted((str(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=True))), sorted((str(_v) for _v in self._components[obj][IVariable] if _active_path_to_object_exists(obj, _v))) if getattr(obj, 'active', True) else [])
        self.assertEqual(set((id(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=True))), set((id(_v) for _v in self._components[obj][IVariable] if _active_path_to_object_exists(obj, _v))) if getattr(obj, 'active', True) else set())
        self.assertEqual(sorted((str(_c) for _c in obj.components(active=True, descend_into=True))), sorted((str(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype] if _active_path_to_object_exists(obj, _c))) if getattr(obj, 'active', True) else [])
        self.assertEqual(set((id(_c) for _c in obj.components(active=True, descend_into=True))), set((id(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype] if _active_path_to_object_exists(obj, _c))) if getattr(obj, 'active', True) else set())