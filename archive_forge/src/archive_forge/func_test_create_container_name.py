from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.put_container')
def test_create_container_name(self, mock_put):
    res_prop = self.t['Resources']['SwiftContainer']['Properties']
    res_prop['name'] = 'the_name'
    stack = utils.parse_stack(self.t)
    container = self._create_container(stack)
    container_name = container.physical_resource_name()
    self.assertEqual('the_name', container_name)
    mock_put.assert_called_once_with('the_name', {})