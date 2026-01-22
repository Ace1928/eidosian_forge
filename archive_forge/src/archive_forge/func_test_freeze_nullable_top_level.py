from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_freeze_nullable_top_level(self):

    class NullFunction(function.Function):

        def result(self):
            return Ellipsis
    null_func = NullFunction(None, 'null', [])
    rd = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties=null_func, metadata=null_func, update_policy=null_func)
    frozen = rd.freeze()
    self.assertIsNone(frozen._properties)
    self.assertIsNone(frozen._metadata)
    self.assertIsNone(frozen._update_policy)
    rd2 = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': null_func, 'Blarg': 'wibble'}, metadata={'Bar': null_func, 'Baz': 'quux'}, update_policy={'some_policy': null_func})
    frozen2 = rd2.freeze()
    self.assertNotIn('Foo', frozen2._properties)
    self.assertEqual('wibble', frozen2._properties['Blarg'])
    self.assertNotIn('Bar', frozen2._metadata)
    self.assertEqual('quux', frozen2._metadata['Baz'])
    self.assertEqual({}, frozen2._update_policy)