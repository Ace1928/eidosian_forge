import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_agent_list_show_set(self):
    """Test network agent list, set, show commands

        Do these serially because show and set rely on the existing agent IDs
        from the list output and we had races when run in parallel.
        """
    agent_list = self.openstack('network agent list', parse_output=True)
    self.assertIsNotNone(agent_list[0])
    agent_ids = list([row['ID'] for row in agent_list])
    cmd_output = self.openstack('network agent show %s' % agent_ids[0], parse_output=True)
    self.assertEqual(agent_ids[0], cmd_output['id'])
    if 'ovn' in agent_list[0]['Agent Type'].lower():
        return
    raw_output = self.openstack('network agent set --disable %s' % agent_ids[0])
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('network agent show %s' % agent_ids[0], parse_output=True)
    self.assertEqual(False, cmd_output['admin_state_up'])
    raw_output = self.openstack('network agent set --enable %s' % agent_ids[0])
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('network agent show %s' % agent_ids[0], parse_output=True)
    self.assertEqual(True, cmd_output['admin_state_up'])