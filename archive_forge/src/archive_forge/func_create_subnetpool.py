from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import subnetpool
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def create_subnetpool(self, status='COMPLETE', tags=None):
    self.t = template_format.parse(inline_templates.SPOOL_TEMPLATE)
    if tags:
        self.t['resources']['sub_pool']['properties']['tags'] = tags
    self.stack = utils.parse_stack(self.t)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    rsrc = subnetpool.SubnetPool('sub_pool', resource_defns['sub_pool'], self.stack)
    if status == 'FAILED':
        self.patchobject(neutronclient.Client, 'create_subnetpool', side_effect=qe.NeutronClientException(status_code=500))
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.sub_pool: An unknown exception occurred.', str(error))
    else:
        self.patchobject(neutronclient.Client, 'create_subnetpool', return_value={'subnetpool': {'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}})
        scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, status), rsrc.state)
    if tags:
        self.set_tag_mock.assert_called_once_with('subnetpools', rsrc.resource_id, {'tags': tags})
    return rsrc