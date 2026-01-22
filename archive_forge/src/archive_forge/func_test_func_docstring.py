import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_func_docstring(self):
    ignore_func = self._make_filter_func()
    self.assertEqual('Ignore some exceptions F.', ignore_func.__doc__)