from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_render_hot(self):
    rd = self.make_me_one_with_everything()
    expected_hot = {'type': 'SomeType', 'properties': {'Foo': {'Fn::Join': ['a', ['b', 'r']]}, 'Blarg': 'wibble'}, 'metadata': {'Baz': {'Fn::Join': ['u', ['q', '', 'x']]}}, 'depends_on': ['other_resource'], 'deletion_policy': 'Retain', 'update_policy': {'SomePolicy': {}}}
    self.assertEqual(expected_hot, rd.render_hot())