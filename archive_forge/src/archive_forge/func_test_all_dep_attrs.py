import itertools
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_all_dep_attrs(self):
    for res in self.stack.values():
        definitions = (self.stack.defn.resource_definition(n) for n in self.parsed_tmpl['resources'])
        attrs = set(itertools.chain.from_iterable((d.dep_attrs(res.name, load_all=True) for d in definitions)))
        self.assertEqual(self.expected[res.name], attrs)