from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_links(self):
    task1 = _task('task1')
    task2 = _task('task2')
    f = gf.Flow('test').add(task1, task2)
    linked = f.link(task1, task2)
    self.assertIs(linked, f)
    self.assertCountEqual(list(f.iter_links()), [(task1, task2, {'manual': True})])