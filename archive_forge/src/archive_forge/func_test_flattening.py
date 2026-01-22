import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_flattening(self):
    f1 = _captured_failure('Wrap me')
    f2 = _captured_failure('Wrap me, too')
    f3 = _captured_failure('Woot!')
    try:
        raise exceptions.WrappedFailure([f1, f2])
    except Exception:
        fail_obj = failure.Failure()
    wf = exceptions.WrappedFailure([fail_obj, f3])
    self.assertEqual([f1, f2, f3], list(wf))