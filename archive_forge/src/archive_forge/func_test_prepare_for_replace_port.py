from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_prepare_for_replace_port(self):
    t = template_format.parse(neutron_port_template)
    stack = utils.parse_stack(t)
    port = stack['port']
    port.resource_id = 'test_res_id'
    _value = {'fixed_ips': {'subnet_id': 'test_subnet', 'ip_address': '42.42.42.42'}}
    port._show_resource = mock.Mock(return_value=_value)
    port.data_set = mock.Mock()
    n_client = mock.Mock()
    port.client = mock.Mock(return_value=n_client)
    port.prepare_for_replace()
    port.data_set.assert_called_once_with('port_fip', jsonutils.dumps(_value.get('fixed_ips')))
    expected_props = {'port': {'fixed_ips': []}}
    n_client.update_port.assert_called_once_with('test_res_id', expected_props)