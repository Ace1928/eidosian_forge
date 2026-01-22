import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_tuple_equals_itself(self):
    exc_info = _make_exc_info('Woot!')
    self.assertTrue(failure._are_equal_exc_info_tuples(exc_info, exc_info))