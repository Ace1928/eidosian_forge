import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_save_as(self):
    flow = utils.TaskOneReturn(name='task1', provides='first_data')
    engine = self._make_engine(flow)
    engine.run()
    self.assertEqual({'first_data': 1}, engine.storage.fetch_all())