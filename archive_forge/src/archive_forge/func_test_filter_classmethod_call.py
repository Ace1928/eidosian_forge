import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_filter_classmethod_call(self):
    ignore_assertion_error = self._make_filter_classmethod()
    try:
        assert False, 'This is a test'
    except Exception as exc:
        ignore_assertion_error(exc)