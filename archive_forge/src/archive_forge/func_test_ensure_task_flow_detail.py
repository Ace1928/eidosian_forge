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
def test_ensure_task_flow_detail(self):
    _lb, flow_detail = p_utils.temporary_flow_detail(self.backend)
    s = self._get_storage(flow_detail)
    t = test_utils.NoopTask('my task')
    t.version = (3, 11)
    s.ensure_atom(t)
    td = flow_detail.find(s.get_atom_uuid('my task'))
    self.assertIsNotNone(td)
    self.assertEqual('my task', td.name)
    self.assertEqual('3.11', td.version)
    self.assertEqual(states.PENDING, td.state)