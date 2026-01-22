import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_raise_previous_after_filtered_func_call(self):
    ignore_assertion_error = self._make_filter_func()
    try:
        raise RuntimeError
    except Exception as exc1:
        try:
            assert False, 'This is a test'
        except Exception:
            pass
        self.assertRaises(RuntimeError, ignore_assertion_error, exc1)