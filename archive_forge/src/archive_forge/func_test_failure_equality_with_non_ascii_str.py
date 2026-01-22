import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_failure_equality_with_non_ascii_str(self):
    bad_string = chr(200)
    fail = failure.Failure.from_exception(ValueError(bad_string))
    copied = fail.copy()
    self.assertEqual(fail, copied)