from openstackclient.tests.functional.compute.v2 import common
def test_server_event_list_and_show(self):
    """Test list, show server event"""
    cmd_output = self.openstack('server event list ' + self.server_name, parse_output=True)
    request_id = None
    for each_event in cmd_output:
        self.assertNotIn('Message', each_event)
        self.assertNotIn('Project ID', each_event)
        self.assertNotIn('User ID', each_event)
        if each_event.get('Action') == 'create':
            self.assertEqual(self.server_id, each_event.get('Server ID'))
            request_id = each_event.get('Request ID')
            break
    self.assertIsNotNone(request_id)
    cmd_output = self.openstack('server event show ' + self.server_name + ' ' + request_id, parse_output=True)
    self.assertEqual(request_id, cmd_output.get('request_id'))
    self.assertEqual('create', cmd_output.get('action'))
    self.assertIsNotNone(cmd_output.get('events'))
    self.assertIsInstance(cmd_output.get('events'), list)
    self.openstack('server reboot --wait ' + self.server_name)
    cmd_output = self.openstack('server event list --long ' + self.server_name, parse_output=True)
    request_id = None
    for each_event in cmd_output:
        self.assertIn('Message', each_event)
        self.assertIn('Project ID', each_event)
        self.assertIn('User ID', each_event)
        if each_event.get('Action') == 'reboot':
            request_id = each_event.get('Request ID')
            self.assertEqual(self.server_id, each_event.get('Server ID'))
            break
    self.assertIsNotNone(request_id)
    cmd_output = self.openstack('server event show ' + self.server_name + ' ' + request_id, parse_output=True)
    self.assertEqual(request_id, cmd_output.get('request_id'))
    self.assertEqual('reboot', cmd_output.get('action'))
    self.assertIsNotNone(cmd_output.get('events'))
    self.assertIsInstance(cmd_output.get('events'), list)