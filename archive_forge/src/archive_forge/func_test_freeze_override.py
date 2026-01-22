from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_freeze_override(self):
    rd = self.make_me_one_with_everything()
    frozen = rd.freeze(metadata={'Baz': 'wibble'})
    self.assertEqual('bar', frozen._properties['Foo'])
    self.assertEqual('wibble', frozen._metadata['Baz'])