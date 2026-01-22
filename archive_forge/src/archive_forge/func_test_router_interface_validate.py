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
def test_router_interface_validate(self):

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        id_mapping = {'router': 'ae478782-53c0-4434-ab16-49900c88016c', 'subnet': '8577cafd-8e98-4059-a2e6-8a771b4d318e', 'port': '9577cafd-8e98-4059-a2e6-8a771b4d318e'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    t = template_format.parse(neutron_template)
    json = t['resources']['router_interface']
    json['properties'] = {'router_id': 'ae478782-53c0-4434-ab16-49900c88016c', 'subnet_id': '8577cafd-8e98-4059-a2e6-8a771b4d318e', 'port_id': '9577cafd-8e98-4059-a2e6-8a771b4d318e'}
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    res = router.RouterInterface('router_interface', resource_defns['router_interface'], stack)
    self.assertRaises(exception.ResourcePropertyConflict, res.validate)
    json['properties'] = {'router_id': 'ae478782-53c0-4434-ab16-49900c88016c', 'port_id': '9577cafd-8e98-4059-a2e6-8a771b4d318e'}
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    res = router.RouterInterface('router_interface', resource_defns['router_interface'], stack)
    self.assertIsNone(res.validate())
    json['properties'] = {'router_id': 'ae478782-53c0-4434-ab16-49900c88016c', 'subnet_id': '8577cafd-8e98-4059-a2e6-8a771b4d318e'}
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    res = router.RouterInterface('router_interface', resource_defns['router_interface'], stack)
    self.assertIsNone(res.validate())
    json['properties'] = {'router_id': 'ae478782-53c0-4434-ab16-49900c88016c'}
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    res = router.RouterInterface('router_interface', resource_defns['router_interface'], stack)
    ex = self.assertRaises(exception.PropertyUnspecifiedError, res.validate)
    self.assertEqual('At least one of the following properties must be specified: subnet, port.', str(ex))