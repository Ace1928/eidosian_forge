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
def test_restore_prev_rsrc(self):
    t = template_format.parse(neutron_port_template)
    stack = utils.parse_stack(t)
    new_port = stack['port']
    new_port.resource_id = 'new_res_id'
    old_port = mock.Mock()
    new_port.stack._backup_stack = mock.Mock()
    new_port.stack._backup_stack().resources.get.return_value = old_port
    old_port.resource_id = 'old_res_id'
    _value = {'subnet_id': 'test_subnet', 'ip_address': '42.42.42.42'}
    old_port.data = mock.Mock(return_value={'port_fip': jsonutils.dumps(_value)})
    n_client = mock.Mock()
    new_port.client = mock.Mock(return_value=n_client)
    new_port.restore_prev_rsrc()
    expected_new_props = {'port': {'fixed_ips': []}}
    expected_old_props = {'port': {'fixed_ips': _value}}
    n_client.update_port.assert_has_calls([mock.call('new_res_id', expected_new_props), mock.call('old_res_id', expected_old_props)])