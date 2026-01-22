from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_retry_and_task_provide_same_value(self):
    flow = gf.Flow('gf', retry.AlwaysRevert('rt', provides=['x']))
    flow.add(utils.TaskOneReturn('t1', provides=['x']))
    self.assertEqual(set(['x']), flow.provides)