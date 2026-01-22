import itertools
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_referenced_attrs_single_output(self):
    self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs={'out1'}), {('flat_dict', 'key2'), ('nested_dict', 'string')})
    self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs={'out1'}), {'attr_B3'})
    self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs={'out2'}), set())
    self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs={'out2'}), set())