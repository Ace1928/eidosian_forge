import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_deletion_policy_titlecase(self):
    hot_tpl = template_format.parse('\n        heat_template_version: 2016-10-14\n        resources:\n          del:\n            type: OS::Heat::None\n            deletion_policy: Delete\n          ret:\n            type: OS::Heat::None\n            deletion_policy: Retain\n          snap:\n            type: OS::Heat::None\n            deletion_policy: Snapshot\n        ')
    rsrc_defns = template.Template(hot_tpl).resource_definitions(None)
    self.assertEqual(rsrc_defn.ResourceDefinition.DELETE, rsrc_defns['del'].deletion_policy())
    self.assertEqual(rsrc_defn.ResourceDefinition.RETAIN, rsrc_defns['ret'].deletion_policy())
    self.assertEqual(rsrc_defn.ResourceDefinition.SNAPSHOT, rsrc_defns['snap'].deletion_policy())