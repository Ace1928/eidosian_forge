from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_stringy(self):
    f = lf.Flow('test')
    expected = '"linear_flow.Flow: test(len=0)"'
    self.assertEqual(expected, str(f))
    task1 = _task(name='task1')
    task2 = _task(name='task2')
    task3 = _task(name='task3')
    f = lf.Flow('test')
    f.add(task1, task2, task3)
    expected = '"linear_flow.Flow: test(len=3)"'
    self.assertEqual(expected, str(f))