from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def make_me_one_with_everything(self):
    return rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': cfn_funcs.Join(None, 'Fn::Join', ['a', ['b', 'r']]), 'Blarg': 'wibble'}, metadata={'Baz': cfn_funcs.Join(None, 'Fn::Join', ['u', ['q', '', 'x']])}, depends=['other_resource'], deletion_policy='Retain', update_policy={'SomePolicy': {}})