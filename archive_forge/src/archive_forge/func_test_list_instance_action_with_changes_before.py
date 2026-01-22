import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_list_instance_action_with_changes_before(self):
    server = self._create_server()
    end_create = self._wait_for_instance_actions(server, 1)
    time.sleep(1)
    server.stop()
    end_stop = self._wait_for_instance_actions(server, 2)
    stop_output = self.nova('instance-action-list %s --changes-before %s' % (server.id, end_stop))
    action = self._get_list_of_values_from_single_column_table(stop_output, 'Action')
    self.assertEqual(['create', 'stop'], action, 'Expected to find the create and stop actions with --changes-before=%s but got: %s\n\n' % (end_stop, stop_output))
    create_output = self.nova('instance-action-list %s --changes-before %s' % (server.id, end_create))
    action = self._get_list_of_values_from_single_column_table(create_output, 'Action')
    self.assertEqual(['create'], action, 'Expected to find the create action with --changes-before=%s but got: %s\n\nFirst instance-action-list output: %s' % (end_create, create_output, stop_output))