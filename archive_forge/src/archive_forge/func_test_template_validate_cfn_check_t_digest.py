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
def test_template_validate_cfn_check_t_digest(self):
    t = {'AWSTemplateFormatVersion': '2010-09-09', 'Description': 'foo', 'Parameters': {}, 'Mappings': {}, 'Resources': {'server': {'Type': 'OS::Nova::Server'}}, 'Outputs': {}}
    tmpl = template.Template(t)
    self.assertIsNone(tmpl.t_digest)
    tmpl.validate()
    self.assertEqual(hashlib.sha256(str(t).encode('utf-8')).hexdigest(), tmpl.t_digest, 'invalid template digest')