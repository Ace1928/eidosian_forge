import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def test_resumption_on_crash_after_revert_scheduled(self):
    engine = self._pretend_to_run_a_flow_and_crash('revert scheduled')
    with utils.CaptureListener(engine) as capturer:
        engine.run()
    expected = ['task1.t REVERTED(None)', 'flow-1_retry.r RETRYING', 'task1.t PENDING', 'flow-1_retry.r RUNNING', 'flow-1_retry.r SUCCESS(2)', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'flow-1.f SUCCESS']
    self.assertEqual(expected, capturer.values)