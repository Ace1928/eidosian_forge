import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_child_template_empty_resource_list(self):
    tmpl_def = copy.deepcopy(TEMPLATE)
    tmpl_def['resources']['test-chain']['properties']['resources'] = []
    chain = self._create_chain(tmpl_def)
    child_template = chain.child_template()
    tmpl = child_template.t
    self.assertNotIn('resources', tmpl)
    self.assertIn('heat_template_version', tmpl)