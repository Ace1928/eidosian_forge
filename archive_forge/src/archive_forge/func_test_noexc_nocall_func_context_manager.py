import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_noexc_nocall_func_context_manager(self):

    @excutils.exception_filter
    def translate_exceptions(ex):
        raise RuntimeError
    with translate_exceptions:
        pass