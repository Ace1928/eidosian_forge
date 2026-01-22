import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_no_capture_exc_args(self):
    captured = _captured_failure(Exception('I am not valid JSON'))
    fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured), exc_args=list(captured.exception_args))
    fail_json = fail_obj.to_dict(include_args=False)
    self.assertNotEqual(fail_obj.exception_args, fail_json['exc_args'])
    self.assertEqual(fail_json['exc_args'], tuple())