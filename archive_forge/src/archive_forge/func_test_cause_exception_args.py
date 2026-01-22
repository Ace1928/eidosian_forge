import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_cause_exception_args(self):
    f = _captured_failure('Woot!')
    d_f = f.to_dict()
    self.assertEqual(1, len(d_f['exc_args']))
    self.assertEqual(('Woot!',), d_f['exc_args'])
    f2 = failure.Failure.from_dict(d_f)
    self.assertEqual(f.exception_args, f2.exception_args)