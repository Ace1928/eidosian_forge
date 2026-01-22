import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_update_instance_type(self):
    """Test case for updating InstanceType.

        Instance.handle_update supports changing the InstanceType, and makes
        the change making a resize API call against Nova.
        """
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    instance = self._create_test_instance(return_server, 'ud_type')
    update_props = self.instance_props.copy()
    update_props['InstanceType'] = 'm1.small'
    update_template = instance.t.freeze(properties=update_props)

    def side_effect(*args):
        return 2 if args[0] == 'm1.small' else 1
    self.patchobject(nova.NovaClientPlugin, 'find_flavor_by_name_or_id', side_effect=side_effect)
    self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=1)
    statuses = iter(['ACTIVE', 'RESIZE', 'VERIFY_RESIZE', 'VERIFY_RESIZE', 'ACTIVE'])

    def get_with_status(*args):
        return_server.status = next(statuses)
        return return_server
    self.fc.servers.get = mock.Mock(side_effect=get_with_status)
    self.fc.client.post_servers_1234_action = mock.Mock(return_value=(202, None))
    scheduler.TaskRunner(instance.update, update_template)()
    self.assertEqual((instance.UPDATE, instance.COMPLETE), instance.state)
    self.fc.servers.get.assert_called_with('1234')
    self.assertEqual(5, self.fc.servers.get.call_count)
    self.fc.client.post_servers_1234_action.assert_has_calls([mock.call(body={'resize': {'flavorRef': 2}}), mock.call(body={'confirmResize': None})])
    self.assertEqual(2, self.fc.client.post_servers_1234_action.call_count)