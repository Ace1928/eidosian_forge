from unittest import mock
from oslo_versionedobjects import exception
from oslo_versionedobjects import test
def test_object_action_error(self):
    exc = exception.ObjectActionError(action='ACTION', reason='REASON', code=123)
    self.assertEqual('Object action ACTION failed because: REASON', str(exc))
    self.assertEqual({'code': 123, 'action': 'ACTION', 'reason': 'REASON'}, exc.kwargs)