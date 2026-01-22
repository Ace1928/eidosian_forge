from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_requires_rebind_values(self):
    flow = uf.Flow('uf').add(utils.TaskOneArg('task1', rebind=['q']), utils.TaskMultiArg('task2'))
    self.assertEqual(set(['x', 'y', 'z', 'q']), flow.requires)
    self.assertEqual(set(), flow.provides)