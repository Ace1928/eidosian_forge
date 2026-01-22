import contextlib
import threading
from oslo_utils import uuidutils
from taskflow import exceptions
from taskflow.persistence import backends
from taskflow.persistence import models
from taskflow import states
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import persistence_utils as p_utils
def test_get_flow_state(self):
    _lb, flow_detail = p_utils.temporary_flow_detail(backend=self.backend)
    flow_detail.state = states.FAILURE
    with contextlib.closing(self.backend.get_connection()) as conn:
        flow_detail.update(conn.update_flow_details(flow_detail))
    s = self._get_storage(flow_detail)
    self.assertEqual(states.FAILURE, s.get_flow_state())