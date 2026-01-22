import gc
from io import StringIO
import weakref
from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
def test_plugin_interface(self):

    class IFoo(Interface):
        pass

    class myFoo(Plugin):
        implements(IFoo)
    ep = ExtensionPoint(IFoo)
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {myFoo: {}})
    self.assertEqual(len(ep), 0)
    a = myFoo()
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), False)}})
    self.assertEqual(len(ep), 0)
    a.activate()
    self.assertEqual(ep.extensions(), [a])
    self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), True)}})
    self.assertEqual(len(ep), 1)
    a.deactivate()
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), False)}})
    self.assertEqual(len(ep), 0)
    a = None
    gc.collect()
    gc.collect()
    gc.collect()
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {myFoo: {}})
    self.assertEqual(len(ep), 0)