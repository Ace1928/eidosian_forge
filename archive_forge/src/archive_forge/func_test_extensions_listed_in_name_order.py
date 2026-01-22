from unittest import mock
from stevedore import named
from stevedore.tests import utils
def test_extensions_listed_in_name_order(self):
    em = named.NamedExtensionManager('stevedore.test.extension', names=['t1', 't2'], name_order=True)
    actual = em.names()
    self.assertEqual(actual, ['t1', 't2'])
    em = named.NamedExtensionManager('stevedore.test.extension', names=['t2', 't1'], name_order=True)
    actual = em.names()
    self.assertEqual(actual, ['t2', 't1'])