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
def test_non_saving_storage(self):
    _lb, flow_detail = p_utils.temporary_flow_detail(self.backend)
    s = storage.Storage(flow_detail=flow_detail)
    s.ensure_atom(test_utils.NoopTask('my_task'))
    self.assertTrue(uuidutils.is_uuid_like(s.get_atom_uuid('my_task')))