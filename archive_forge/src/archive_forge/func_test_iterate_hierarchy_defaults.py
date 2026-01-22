from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_iterate_hierarchy_defaults(self):
    self.assertIs(TraversalStrategy.BFS, TraversalStrategy.BreadthFirstSearch)
    self.assertIs(TraversalStrategy.DFS, TraversalStrategy.PrefixDepthFirstSearch)
    self.assertIs(TraversalStrategy.DFS, TraversalStrategy.PrefixDFS)
    self.assertIs(TraversalStrategy.DFS, TraversalStrategy.ParentFirstDepthFirstSearch)
    self.assertIs(TraversalStrategy.PostfixDepthFirstSearch, TraversalStrategy.PostfixDFS)
    self.assertIs(TraversalStrategy.PostfixDepthFirstSearch, TraversalStrategy.ParentLastDepthFirstSearch)
    HM = HierarchicalModel()
    m = HM.model
    result = [x.name for x in m.block_data_objects()]
    self.assertEqual(HM.PrefixDFS, result)