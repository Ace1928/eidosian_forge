import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_causes_suppress_context(self):
    f = None
    try:
        try:
            self._raise_many(['Still still not working', 'Still not working', 'Not working'])
        except RuntimeError as e:
            raise e from None
    except RuntimeError:
        f = failure.Failure()
    self.assertIsNotNone(f)
    self.assertEqual([], list(f.causes))