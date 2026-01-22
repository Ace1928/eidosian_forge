from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_checks_for_dups(self):
    flo = gf.Flow('test').add(test_utils.DummyTask(name='a'), test_utils.DummyTask(name='a'))
    e = engines.load(flo)
    self.assertRaisesRegex(exc.Duplicate, '^Atoms with duplicate names', e.compile)