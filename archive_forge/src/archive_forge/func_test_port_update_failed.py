import copy
from unittest import mock
from ironicclient.common.apiclient import exceptions as ic_exc
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import ironic as ic
from heat.engine import resource
from heat.engine.resources.openstack.ironic import port
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_port_update_failed(self):
    exc_msg = 'Port 9cc6fd32-f711-4e1f-a82d-59e6ae074e95 can not have any connectivity attributes (pxe_enabled, portgroup_id, physical_network, local_link_connection) updated unless node 9ccee9ec-92a5-4580-9242-82eb7f454d3f is in a enroll, inspecting, inspect wait, manageable state or in maintenance mode.'
    self._port_update(exc_msg)