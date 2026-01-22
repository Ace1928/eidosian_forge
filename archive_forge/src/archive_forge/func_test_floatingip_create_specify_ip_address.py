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
def test_floatingip_create_specify_ip_address(self):
    self.stub_NetworkConstraint_validate()
    self.mockclient.create_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_ip_address': '172.24.4.98'}}
    self.mockclient.show_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_ip_address': '172.24.4.98'}}
    t = template_format.parse(neutron_floating_template)
    props = t['resources']['floating_ip']['properties']
    props['floating_ip_address'] = '172.24.4.98'
    stack = utils.parse_stack(t)
    fip = stack['floating_ip']
    scheduler.TaskRunner(fip.create)()
    self.assertEqual((fip.CREATE, fip.COMPLETE), fip.state)
    self.assertEqual('172.24.4.98', fip.FnGetAtt('floating_ip_address'))
    self.mockclient.create_floatingip.assert_called_once_with({'floatingip': {'floating_network_id': u'abcd1234', 'floating_ip_address': '172.24.4.98'}})
    self.mockclient.show_floatingip.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')