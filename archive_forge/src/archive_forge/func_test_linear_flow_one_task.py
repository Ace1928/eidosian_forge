from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_one_task(self):
    f = lf.Flow('test')
    task = _task(name='task1', requires=['a', 'b'], provides=['c', 'd'])
    result = f.add(task)
    self.assertIs(f, result)
    self.assertEqual(1, len(f))
    self.assertEqual([task], list(f))
    self.assertEqual([], list(f.iter_links()))
    self.assertEqual(set(['a', 'b']), f.requires)
    self.assertEqual(set(['c', 'd']), f.provides)