from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_two_dependent_tasks_two_different_calls(self):
    task1 = _task(name='task1', provides=['a'])
    task2 = _task(name='task2', requires=['a'])
    f = gf.Flow('test').add(task1).add(task2)
    self.assertEqual(2, len(f))
    self.assertCountEqual(f, [task1, task2])
    self.assertEqual([(task1, task2, {'reasons': set(['a'])})], list(f.iter_links()))