import itertools
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_referenced_attrs_both(self):
    self.assertEqual(self.resA.referenced_attrs(in_resources=True, in_outputs=True), {('list', 1), ('nested_dict', 'dict', 'b'), ('flat_dict', 'key2'), ('nested_dict', 'string')})
    self.assertEqual(self.resB.referenced_attrs(in_resources=True, in_outputs=True), {'attr_B3'})