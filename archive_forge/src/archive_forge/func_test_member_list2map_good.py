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
def test_member_list2map_good(self):
    tmpl = template.Template(empty_template)
    snippet = {'Fn::MemberListToMap': ['Name', 'Value', ['.member.0.Name=metric', '.member.0.Value=cpu', '.member.1.Name=size', '.member.1.Value=56']]}
    self.assertEqual({'metric': 'cpu', 'size': '56'}, self.resolve(snippet, tmpl))