import copy
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def mock_show_subnet(self):
    self.m_ss.return_value = {'subnet': {'name': 'my_subnet', 'network_id': 'nnnn', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'allocation_pools': [{'start': '10.0.0.2', 'end': '10.0.0.254'}], 'gateway_ip': '10.0.0.1', 'ip_version': 4, 'cidr': '10.0.0.0/24', 'id': 'ssss', 'enable_dhcp': False}}