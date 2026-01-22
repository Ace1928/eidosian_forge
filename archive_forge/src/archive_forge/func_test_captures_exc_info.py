import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_captures_exc_info(self):
    exc_info = self.fail_obj.exc_info
    self.assertEqual(3, len(exc_info))
    self.assertEqual(RuntimeError, exc_info[0])
    self.assertIs(exc_info[1], self.fail_obj.exception)