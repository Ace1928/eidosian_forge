import copy
from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception as heat_ex
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.nova import floatingip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def mock_interface(self, port, ip):

    class MockIface(object):

        def __init__(self, port_id, fixed_ip):
            self.port_id = port_id
            self.fixed_ips = [{'ip_address': fixed_ip}]
    return MockIface(port, ip)