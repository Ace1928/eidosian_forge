from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_service import threadgroup
from swiftclient import exceptions
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_stack_create_with_container_notfound_swift(self):
    stack_name = 'env_files_test_stack'
    environment_files = ['env_1', 'env_2']
    files_container = 'test_container'
    mock_client = mock.Mock()
    mock_client.get_container.side_effect = exceptions.ClientException('error')
    self.patchobject(swift.SwiftClientPlugin, '_create', return_value=mock_client)
    self._test_stack_create(stack_name, environment_files=environment_files, files_container=files_container, error=True)
    mock_client.get_container.assert_called_with(files_container)
    mock_client.get_object.assert_not_called()