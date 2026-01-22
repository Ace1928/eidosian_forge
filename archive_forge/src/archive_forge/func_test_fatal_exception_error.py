from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_fatal_exception_error(self):
    self.useFixture(fixtures.MonkeyPatch('heat.common.exception._FATAL_EXCEPTION_FORMAT_ERRORS', True))
    self.assertRaises(KeyError, TestException)