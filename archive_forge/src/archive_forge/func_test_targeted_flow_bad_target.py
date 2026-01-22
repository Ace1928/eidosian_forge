from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_targeted_flow_bad_target(self):
    f = gf.TargetedFlow('test')
    task1 = _task('task1', provides=['a'], requires=[])
    task2 = _task('task2', provides=['b'], requires=['a'])
    f.add(task1)
    self.assertRaisesRegex(ValueError, '^Node .* not found', f.set_target, task2)