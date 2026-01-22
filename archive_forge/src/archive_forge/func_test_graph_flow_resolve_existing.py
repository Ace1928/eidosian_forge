from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_resolve_existing(self):
    task1 = _task(name='task1', requires=['a', 'b'])
    task2 = _task(name='task2', provides=['a', 'b'])
    f = gf.Flow('test')
    f.add(task1)
    f.add(task2, resolve_existing=True)
    self.assertEqual(set([]), f.requires)