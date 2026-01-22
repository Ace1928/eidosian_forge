from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_add_nothing(self):
    f = uf.Flow('test')
    result = f.add()
    self.assertIs(f, result)
    self.assertEqual(0, len(f))