from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_nested_flows_requirements(self):
    flow = uf.Flow('uf').add(lf.Flow('lf').add(utils.TaskOneArgOneReturn('task1', rebind=['a'], provides=['x']), utils.TaskOneArgOneReturn('task2', provides=['y'])), uf.Flow('uf').add(utils.TaskOneArgOneReturn('task3', rebind=['b'], provides=['z']), utils.TaskOneArgOneReturn('task4', rebind=['c'], provides=['q'])))
    self.assertEqual(set(['a', 'b', 'c']), flow.requires)
    self.assertEqual(set(['x', 'y', 'z', 'q']), flow.provides)