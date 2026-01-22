import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_argument_injection_required(self):
    flow = utils.TaskMultiArgOneReturn(provides='result', requires=['a', 'b', 'c'], inject={'x': 1, 'y': 4, 'z': 9, 'a': 0, 'b': 0, 'c': 0})
    engine = self._make_engine(flow)
    engine.run()
    self.assertEqual({'result': 14}, engine.storage.fetch_all())