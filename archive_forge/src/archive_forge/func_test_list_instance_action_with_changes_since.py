import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_list_instance_action_with_changes_since(self):
    before_create = timeutils.utcnow().replace(microsecond=0).isoformat()
    server = self._create_server()
    time.sleep(2)
    before_stop = timeutils.utcnow().replace(microsecond=0).isoformat()
    server.stop()
    create_output = self.nova('instance-action-list %s --changes-since %s' % (server.id, before_create))
    action = self._get_list_of_values_from_single_column_table(create_output, 'Action')
    self.assertEqual(action, ['create', 'stop'])
    stop_output = self.nova('instance-action-list %s --changes-since %s' % (server.id, before_stop))
    action = self._get_list_of_values_from_single_column_table(stop_output, 'Action')
    self.assertEqual(action, ['stop'], 'Expected to find the stop action with --changes-since=%s but got: %s\n\nFirst instance-action-list output: %s' % (before_stop, stop_output, create_output))