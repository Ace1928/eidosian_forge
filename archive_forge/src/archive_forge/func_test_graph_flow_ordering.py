from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_ordering(self):
    task1 = _task('task1', provides=set(['a', 'b']))
    task2 = _task('task2', provides=['c'], requires=['a', 'b'])
    task3 = _task('task3', provides=[], requires=['c'])
    f = gf.Flow('test').add(task1, task2, task3)
    self.assertEqual(3, len(f))
    self.assertCountEqual(list(f.iter_links()), [(task1, task2, {'reasons': set(['a', 'b'])}), (task2, task3, {'reasons': set(['c'])})])