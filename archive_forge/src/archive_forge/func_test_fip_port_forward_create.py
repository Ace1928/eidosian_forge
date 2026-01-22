import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_fip_port_forward_create(self):
    pfid = mock.Mock(id='180941c5-9e82-41c7-b64d-6a57302ec211')
    props = {'internal_ip_address': '10.0.0.10', 'internal_port_number': 8080, 'external_port': 80, 'internal_port': '9c1eb3fe-7bba-479d-bd43-fdb0bc7cd151', 'protocol': 'tcp'}
    mock_create = self.patchobject(self.sdkclient.network, 'create_floating_ip_port_forwarding', return_value=pfid)
    self.mockclient.create_port.return_value = {'port': {'status': 'BUILD', 'id': '9c1eb3fe-7bba-479d-bd43-fdb0bc7cd151'}}
    self.mockclient.show_port.return_value = {'port': {'status': 'ACTIVE', 'id': '9c1eb3fe-7bba-479d-bd43-fdb0bc7cd151'}}
    self.mockclient.create_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': '477e8273-60a7-4c41-b683-1d497e53c384'}}
    p = self.stack['port_floating']
    scheduler.TaskRunner(p.create)()
    self.assertEqual((p.CREATE, p.COMPLETE), p.state)
    stk_defn.update_resource_data(self.stack.defn, p.name, p.node_data())
    fip = self.stack['floating_ip']
    scheduler.TaskRunner(fip.create)()
    self.assertEqual((fip.CREATE, fip.COMPLETE), fip.state)
    stk_defn.update_resource_data(self.stack.defn, fip.name, fip.node_data())
    port_forward = self.stack['port_forwarding']
    scheduler.TaskRunner(port_forward.create)()
    self.assertEqual((port_forward.CREATE, port_forward.COMPLETE), port_forward.state)
    mock_create.assert_called_once_with('477e8273-60a7-4c41-b683-1d497e53c384', **props)