import importlib
import inspect
from unittest import mock
import stevedore
from stevedore import extension
from novaclient import client
from novaclient.tests.unit import utils
def mock_mgr():
    fake_ep = mock.Mock()
    fake_ep.name = 'foo'
    module_spec = importlib.machinery.ModuleSpec('foo', None)
    fake_ep.module = importlib.util.module_from_spec(module_spec)
    fake_ep.load.return_value = fake_ep.module
    fake_ext = extension.Extension(name='foo', entry_point=fake_ep, plugin=fake_ep.module, obj=None)
    return stevedore.ExtensionManager.make_test_instance([fake_ext])