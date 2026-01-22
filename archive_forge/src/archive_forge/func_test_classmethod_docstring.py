import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_classmethod_docstring(self):
    ignore_func = self._make_filter_classmethod()
    self.assertEqual('Ignore some exceptions C.', ignore_func.__doc__)