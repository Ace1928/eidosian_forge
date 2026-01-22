from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_without_dependencies(self):
    flow = uf.Flow('uf').add(utils.TaskNoRequiresNoReturns('task1'), utils.TaskNoRequiresNoReturns('task2'))
    self.assertEqual(set(), flow.requires)
    self.assertEqual(set(), flow.provides)