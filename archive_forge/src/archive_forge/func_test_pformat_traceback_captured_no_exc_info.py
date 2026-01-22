import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_pformat_traceback_captured_no_exc_info(self):
    captured = _captured_failure('Woot!')
    captured = failure.Failure.from_dict(captured.to_dict())
    text = captured.pformat(traceback=True)
    self.assertIn('Traceback (most recent call last):', text)