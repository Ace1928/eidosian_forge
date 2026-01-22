from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_add_nothing(self):
    f = lf.Flow('test')
    result = f.add()
    self.assertIs(f, result)
    self.assertEqual(0, len(f))