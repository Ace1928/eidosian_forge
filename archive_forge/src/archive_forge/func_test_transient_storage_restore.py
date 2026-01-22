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
def test_transient_storage_restore(self):
    _lb, flow_detail = p_utils.temporary_flow_detail(self.backend)
    s = self._get_storage(flow_detail=flow_detail)
    s.inject([('a', 'b')], transient=True)
    s.inject([('b', 'c')])
    s2 = self._get_storage(flow_detail=flow_detail)
    results = s2.fetch_all()
    self.assertEqual({'b': 'c'}, results)