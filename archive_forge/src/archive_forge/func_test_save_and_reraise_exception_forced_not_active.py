import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_save_and_reraise_exception_forced_not_active(self):
    e = excutils.save_and_reraise_exception()
    self.assertRaises(RuntimeError, e.force_reraise)
    e = excutils.save_and_reraise_exception()
    e.capture(check=False)
    self.assertRaises(RuntimeError, e.force_reraise)