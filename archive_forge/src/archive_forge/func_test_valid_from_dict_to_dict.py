import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_valid_from_dict_to_dict(self):
    f = _captured_failure('Woot!')
    d_f = f.to_dict()
    failure.Failure.validate(d_f)
    f2 = failure.Failure.from_dict(d_f)
    self.assertTrue(f.matches(f2))