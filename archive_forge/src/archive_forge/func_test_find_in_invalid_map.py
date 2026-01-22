import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_find_in_invalid_map(self):
    tmpl = template.Template(mapping_template)
    stk = stack.Stack(self.ctx, 'test', tmpl)
    finds = ({'Fn::FindInMap': ['InvalidMapping', 'ValueList', 'foo']}, {'Fn::FindInMap': ['InvalidMapping', 'ValueString', 'baz']}, {'Fn::FindInMap': ['MapList', 'foo', 'bar']}, {'Fn::FindInMap': ['MapString', 'foo', 'bar']})
    for find in finds:
        self.assertRaises((KeyError, TypeError), self.resolve, find, tmpl, stk)