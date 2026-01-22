from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_response_mixed_invalid(self):
    msg = pr.Response(pr.EVENT, details={'progress': 0.5}, event_type='blah', result=1)
    self.assertRaises(excp.InvalidFormat, pr.Response.validate, msg)