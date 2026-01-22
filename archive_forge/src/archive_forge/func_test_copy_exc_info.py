import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_copy_exc_info(self):
    exc_info = _make_exc_info('Woot!')
    result = failure._copy_exc_info(exc_info)
    self.assertIsNot(result, exc_info)
    self.assertIs(result[0], RuntimeError)
    self.assertIsNot(result[1], exc_info[1])
    self.assertIs(result[2], exc_info[2])