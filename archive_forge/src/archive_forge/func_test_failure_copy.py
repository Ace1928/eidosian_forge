import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_failure_copy(self):
    fail_obj = _captured_failure('Woot!')
    copied = fail_obj.copy()
    self.assertIsNot(fail_obj, copied)
    self.assertEqual(fail_obj, copied)
    self.assertTrue(fail_obj.matches(copied))