import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
def test_root_enabled_history(self):
    self.management.api.client.get = mock.Mock(return_value=('resp', None))
    self.assertRaises(Exception, self.management.root_enabled_history, 'instance')
    body = {'root_history': 'rh'}
    self.management.api.client.get = mock.Mock(return_value=('resp', body))
    management.RootHistory.__init__ = mock.Mock(return_value=None)
    rh = self.management.root_enabled_history('instance')
    self.assertIsInstance(rh, management.RootHistory)