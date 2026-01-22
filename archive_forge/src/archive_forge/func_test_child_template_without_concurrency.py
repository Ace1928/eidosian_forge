import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_child_template_without_concurrency(self):
    chain = self._create_chain(TEMPLATE)
    child_template = chain.child_template()
    tmpl = child_template.t
    self.assertEqual('2015-04-30', tmpl['heat_template_version'])
    self.assertEqual(2, len(child_template.t['resources']))
    resource = tmpl['resources']['0']
    self.assertEqual('OS::Heat::SoftwareConfig', resource['type'])
    self.assertEqual(RESOURCE_PROPERTIES, resource['properties'])
    self.assertNotIn('depends_on', resource)
    resource = tmpl['resources']['1']
    self.assertEqual('OS::Heat::StructuredConfig', resource['type'])
    self.assertEqual(RESOURCE_PROPERTIES, resource['properties'])
    self.assertEqual(['0'], resource['depends_on'])