import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_router_validate_ha_distribute(self):
    t = template_format.parse(neutron_template)
    props = t['resources']['router']['properties']
    props['ha'] = True
    props['distributed'] = True
    stack = utils.parse_stack(t)
    rsrc = stack['router']
    update_props = props.copy()
    del update_props['l3_agent_ids']
    rsrc.t = rsrc.t.freeze(properties=update_props)
    rsrc.reparse()
    exc = self.assertRaises(exception.ResourcePropertyConflict, rsrc.validate)
    self.assertIn('distributed, ha', str(exc))