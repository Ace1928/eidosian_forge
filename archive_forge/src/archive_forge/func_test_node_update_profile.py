import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import node as sn
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_node_update_profile(self):
    node = self._create_node()
    self.senlin_mock.get_profile.side_effect = [mock.Mock(id='new_profile_id'), mock.Mock(id='fake_profile_id'), mock.Mock(id='new_profile_id')]
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-node']['properties']
    props['profile'] = 'new_profile'
    props['name'] = 'new_name'
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_node = rsrc_defns['senlin-node']
    self.senlin_mock.update_node.return_value = mock.Mock(location='/actions/fake-action')
    scheduler.TaskRunner(node.update, new_node)()
    self.assertEqual((node.UPDATE, node.COMPLETE), node.state)
    node_update_kwargs = {'profile_id': 'new_profile_id', 'name': 'new_name'}
    self.senlin_mock.update_node.assert_called_once_with(node=self.fake_node, **node_update_kwargs)
    self.assertEqual(2, self.senlin_mock.get_action.call_count)