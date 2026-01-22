from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_nested_exceptions(self):
    res = mock.Mock()
    res.name = 'frodo'
    res.stack.t.get_section_name.return_value = 'Resources'
    reason = 'Resource UPDATE failed: ValueError: resources.oops: Test Resource failed oops'
    base_exc = exception.ResourceFailure(reason, res, action='UPDATE')
    exc = exception.ResourceFailure(base_exc, res, action='UPDATE')
    self.assertEqual(['Resources', 'frodo', 'resources', 'oops'], exc.path)
    self.assertEqual('ValueError', exc.error)
    self.assertEqual('Test Resource failed oops', exc.error_message)