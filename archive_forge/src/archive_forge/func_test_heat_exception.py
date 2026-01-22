from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_heat_exception(self):
    base_exc = ValueError('sorry mom')
    heat_exc = exception.ResourceFailure(base_exc, None, action='UPDATE')
    exc = exception.ResourceFailure(heat_exc, None, action='UPDATE')
    self.assertEqual('ValueError', exc.error)
    self.assertEqual([], exc.path)
    self.assertEqual('sorry mom', exc.error_message)