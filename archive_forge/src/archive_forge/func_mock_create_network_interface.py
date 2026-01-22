import copy
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def mock_create_network_interface(self, stack_name='my_stack', resource_name='my_nic', security_groups=None):
    self.nic_name = utils.PhysName(stack_name, resource_name)
    self.port = {'network_id': 'nnnn', 'fixed_ips': [{'subnet_id': u'ssss'}], 'name': self.nic_name, 'admin_state_up': True}
    port_info = {'port': {'admin_state_up': True, 'device_id': '', 'device_owner': '', 'fixed_ips': [{'ip_address': '10.0.0.100', 'subnet_id': 'ssss'}], 'id': 'pppp', 'mac_address': 'fa:16:3e:25:32:5d', 'name': self.nic_name, 'network_id': 'nnnn', 'status': 'ACTIVE', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f'}}
    if security_groups is not None:
        self.port['security_groups'] = security_groups
        port_info['security_groups'] = security_groups
    else:
        port_info['security_groups'] = ['default']
    self.m_cp.return_value = port_info