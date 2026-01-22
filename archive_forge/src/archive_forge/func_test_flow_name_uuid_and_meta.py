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
def test_flow_name_uuid_and_meta(self):
    flow_detail = models.FlowDetail(name='test-fd', uuid='aaaa')
    flow_detail.meta = {'a': 1}
    s = self._get_storage(flow_detail)
    self.assertEqual('test-fd', s.flow_name)
    self.assertEqual('aaaa', s.flow_uuid)
    self.assertEqual({'a': 1}, s.flow_meta)