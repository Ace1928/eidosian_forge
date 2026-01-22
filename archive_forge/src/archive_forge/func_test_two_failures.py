import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_two_failures(self):
    fls = [_captured_failure('Woot!'), _captured_failure('Oh, not again!')]
    wf = exceptions.WrappedFailure(fls)
    self.assertEqual(2, len(wf))
    self.assertEqual(fls, list(wf))