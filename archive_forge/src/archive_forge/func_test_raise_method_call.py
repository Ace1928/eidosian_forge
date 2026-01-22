import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_raise_method_call(self):
    ignore_assertion_error = self._make_filter_method()
    try:
        raise RuntimeError
    except Exception as exc:
        self.assertRaises(RuntimeError, ignore_assertion_error, exc)