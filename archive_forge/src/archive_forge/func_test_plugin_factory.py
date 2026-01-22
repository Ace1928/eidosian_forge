import gc
from io import StringIO
import weakref
from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
def test_plugin_factory(self):

    class IFoo(Interface):
        pass
    ep = ExtensionPoint(IFoo)

    class myFoo(Plugin):
        implements(IFoo)
        alias('my_foo', 'myFoo docs')
    factory = PluginFactory(IFoo)
    self.assertEqual(factory.services(), ['my_foo'])
    self.assertIsInstance(factory('my_foo'), myFoo)
    self.assertIsNone(factory('unknown'), None)
    self.assertEqual(factory.doc('my_foo'), 'myFoo docs')
    self.assertEqual(factory.doc('unknown'), '')
    self.assertIs(factory.get_class('my_foo'), myFoo)
    self.assertIsNone(factory.get_class('unknown'))
    a = myFoo()
    b = myFoo()
    self.assertFalse(a.enabled())
    self.assertFalse(b.enabled())
    self.assertIsNone(ep.service())
    a.activate()
    self.assertTrue(a.enabled())
    self.assertFalse(b.enabled())
    self.assertIs(ep.service(), a)
    factory.deactivate('my_foo')
    self.assertFalse(a.enabled())
    self.assertFalse(b.enabled())
    self.assertIsNone(ep.service())
    b.activate()
    self.assertFalse(a.enabled())
    self.assertTrue(b.enabled())
    self.assertIs(ep.service(), b)
    gc.collect()
    gc.collect()
    gc.collect()
    factory.activate('my_foo')
    self.assertTrue(a.enabled())
    self.assertTrue(b.enabled())
    with self.assertRaisesRegex(PluginError, "The ExtensionPoint does not have a unique service!  2 services are defined for interface 'IFoo' \\(key=None\\)."):
        self.assertIsNone(ep.service())
    a.deactivate()
    self.assertFalse(a.enabled())
    self.assertTrue(b.enabled())
    self.assertIs(ep.service(), b)
    factory.activate('unknown')
    self.assertFalse(a.enabled())
    self.assertTrue(b.enabled())
    self.assertIs(ep.service(), b)
    factory.deactivate('unknown')
    self.assertFalse(a.enabled())
    self.assertTrue(b.enabled())
    self.assertIs(ep.service(), b)