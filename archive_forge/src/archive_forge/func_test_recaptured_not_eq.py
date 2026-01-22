import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_recaptured_not_eq(self):
    captured = _captured_failure('Woot!')
    fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured), exc_args=list(captured.exception_args))
    self.assertFalse(fail_obj == captured)
    self.assertTrue(fail_obj != captured)
    self.assertTrue(fail_obj.matches(captured))