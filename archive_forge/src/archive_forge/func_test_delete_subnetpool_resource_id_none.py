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
def test_delete_subnetpool_resource_id_none(self):
    delete_pool = self.patchobject(neutronclient.Client, 'delete_subnetpool')
    rsrc = self.create_subnetpool()
    rsrc.resource_id = None
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    delete_pool.assert_not_called()