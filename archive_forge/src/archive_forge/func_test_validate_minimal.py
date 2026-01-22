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
def test_validate_minimal(self):
    self.t = template_format.parse(inline_templates.SPOOL_MINIMAL_TEMPLATE)
    self.stack = utils.parse_stack(self.t)
    rsrc = self.stack['sub_pool']
    self.assertIsNone(rsrc.validate())