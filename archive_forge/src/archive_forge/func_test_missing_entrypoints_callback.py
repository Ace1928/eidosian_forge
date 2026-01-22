from unittest import mock
from testtools.matchers import GreaterThan
from stevedore import extension
from stevedore import named
from stevedore.tests import utils
@mock.patch('stevedore.named.NamedExtensionManager._load_plugins')
def test_missing_entrypoints_callback(self, load_fn):
    errors = set()

    def callback(names):
        errors.update(names)
    load_fn.return_value = [extension.Extension('foo', None, None, None)]
    named.NamedExtensionManager('stevedore.test.extension', names=['foo', 'bar'], invoke_on_load=True, on_missing_entrypoints_callback=callback)
    self.assertEqual(errors, {'bar'})