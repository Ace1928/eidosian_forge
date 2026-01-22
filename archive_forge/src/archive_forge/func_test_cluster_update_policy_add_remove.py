import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import cluster as sc
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_cluster_update_policy_add_remove(self):
    cluster = self._create_cluster(self.t)
    self.senlin_mock.get_policy.side_effect = [mock.Mock(id='new_policy_id'), mock.Mock(id='fake_policy_id'), mock.Mock(id='new_policy_id')]
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-cluster']['properties']
    props['policies'] = [{'policy': 'new_policy'}]
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_cluster = rsrc_defns['senlin-cluster']
    self.senlin_mock.detach_policy_from_cluster.return_value = {'action': 'fake-action'}
    self.senlin_mock.attach_policy_to_cluster.return_value = {'action': 'fake-action'}
    self.senlin_mock.get_action.return_value = mock.Mock(status='SUCCEEDED')
    scheduler.TaskRunner(cluster.update, new_cluster)()
    self.assertEqual((cluster.UPDATE, cluster.COMPLETE), cluster.state)
    detach_policy_kwargs = {'policy': 'fake_policy_id', 'cluster': cluster.resource_id, 'enabled': True}
    self.assertEqual(2, self.senlin_mock.attach_policy_to_cluster.call_count)
    self.senlin_mock.detach_policy_from_cluster.assert_called_once_with(**detach_policy_kwargs)
    self.assertEqual(0, self.senlin_mock.update_cluster_policy.call_count)
    self.assertEqual(3, self.senlin_mock.get_action.call_count)