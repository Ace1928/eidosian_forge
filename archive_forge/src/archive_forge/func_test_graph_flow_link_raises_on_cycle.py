from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_link_raises_on_cycle(self):
    task1 = _task('task1', provides=['a'])
    task2 = _task('task2', requires=['a'])
    f = gf.Flow('test').add(task1, task2)
    self.assertRaises(exc.DependencyFailure, f.link, task2, task1)