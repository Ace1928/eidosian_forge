import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_simple_iter(self):
    fail_obj = _captured_failure('Woot!')
    wf = exceptions.WrappedFailure([fail_obj])
    self.assertEqual(1, len(wf))
    self.assertEqual([fail_obj], list(wf))