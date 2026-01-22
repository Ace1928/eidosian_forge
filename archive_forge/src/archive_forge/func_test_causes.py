import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_causes(self):
    f = None
    try:
        self._raise_many(['Still still not working', 'Still not working', 'Not working'])
    except RuntimeError:
        f = failure.Failure()
    self.assertIsNotNone(f)
    self.assertEqual(2, len(f.causes))
    self.assertEqual('Still not working', f.causes[0].exception_str)
    self.assertEqual('Not working', f.causes[1].exception_str)
    f = f.causes[0]
    self.assertEqual(1, len(f.causes))
    self.assertEqual('Not working', f.causes[0].exception_str)
    f = f.causes[0]
    self.assertEqual(0, len(f.causes))