import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
@mock.patch.object(service_objects.Service, 'active_service_count')
def test_child_template_with_concurrent(self, mock_count):
    tmpl_def = copy.deepcopy(TEMPLATE)
    tmpl_def['resources']['test-chain']['properties']['concurrent'] = True
    chain = self._create_chain(tmpl_def)
    mock_count.return_value = 5
    child_template = chain.child_template()
    tmpl = child_template.t
    resource = tmpl['resources']['0']
    self.assertNotIn('depends_on', resource)
    resource = tmpl['resources']['1']
    self.assertNotIn('depends_on', resource)