import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_no_exc_info(self):
    self.assertIsNone(self.fail_obj.exc_info)