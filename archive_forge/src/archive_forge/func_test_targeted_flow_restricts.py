from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_targeted_flow_restricts(self):
    f = gf.TargetedFlow('test')
    task1 = _task('task1', provides=['a'], requires=[])
    task2 = _task('task2', provides=['b'], requires=['a'])
    task3 = _task('task3', provides=[], requires=['b'])
    task4 = _task('task4', provides=[], requires=['b'])
    f.add(task1, task2, task3, task4)
    f.set_target(task3)
    self.assertEqual(3, len(f))
    self.assertCountEqual(f, [task1, task2, task3])
    self.assertNotIn('c', f.provides)