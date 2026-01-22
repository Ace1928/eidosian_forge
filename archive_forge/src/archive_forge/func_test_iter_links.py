from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_iter_links(self):
    task1 = _task('task1')
    task2 = _task('task2')
    task3 = _task('task3')
    f1 = gf.Flow('nested')
    f1.add(task3)
    tasks = set([task1, task2, f1])
    f = gf.Flow('test').add(task1, task2, f1)
    for u, v, data in f.iter_links():
        self.assertIn(u, tasks)
        self.assertIn(v, tasks)
        self.assertDictEqual({}, data)