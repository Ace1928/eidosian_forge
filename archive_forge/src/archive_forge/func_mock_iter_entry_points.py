import inspect
import types
from unittest import mock
import pkg_resources
import testtools
import troveclient.shell
def mock_iter_entry_points(group):
    if group == 'troveclient.extension':
        fake_ep = mock.Mock()
        fake_ep.name = 'foo'
        fake_ep.module = types.ModuleType('foo')
        fake_ep.load.return_value = fake_ep.module
        return [fake_ep]