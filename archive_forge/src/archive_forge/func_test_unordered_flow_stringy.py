from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_stringy(self):
    f = uf.Flow('test')
    expected = '"unordered_flow.Flow: test(len=0)"'
    self.assertEqual(expected, str(f))
    task1 = _task(name='task1')
    task2 = _task(name='task2')
    task3 = _task(name='task3')
    f = uf.Flow('test')
    f.add(task1, task2, task3)
    expected = '"unordered_flow.Flow: test(len=3)"'
    self.assertEqual(expected, str(f))