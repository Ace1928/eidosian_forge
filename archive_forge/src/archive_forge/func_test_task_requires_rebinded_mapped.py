from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_task_requires_rebinded_mapped(self):
    flow = utils.TaskMultiArg(rebind={'x': 'a', 'y': 'b', 'z': 'c'})
    self.assertEqual(set(['a', 'b', 'c']), flow.requires)
    self.assertEqual(set(), flow.provides)