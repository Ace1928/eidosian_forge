import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_save_and_reraise_exception(self):
    e = None
    msg = 'foo'
    try:
        try:
            raise Exception(msg)
        except Exception:
            with excutils.save_and_reraise_exception():
                pass
    except Exception as _e:
        e = _e
    self.assertEqual(str(e), msg)