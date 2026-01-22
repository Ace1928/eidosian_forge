import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_argument_injection_rebind(self):
    flow = utils.TaskMultiArgOneReturn(provides='result', rebind=['a', 'b', 'c'], inject={'a': 1, 'b': 4, 'c': 9})
    engine = self._make_engine(flow)
    engine.run()
    self.assertEqual({'result': 14}, engine.storage.fetch_all())