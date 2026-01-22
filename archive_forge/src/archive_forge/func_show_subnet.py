import copy
import uuid
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources.aws.ec2 import network_interface as net_interfaces
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def show_subnet(self, subnet, **_params):
    return {'subnet': {'name': 'name', 'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'allocation_pools': [{'start': '10.10.0.2', 'end': '10.10.0.254'}], 'gateway_ip': '10.10.0.1', 'ip_version': 4, 'cidr': '10.10.0.0/24', 'id': '4156c7a5-e8c4-4aff-a6e1-8f3c7bc83861', 'enable_dhcp': False}}