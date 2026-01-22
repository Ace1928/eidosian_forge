import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def test_show_host_with_name_startwith_number(self):
    list_value = [{'id': '101', 'hypervisor_hostname': '1-host'}, {'id': '201', 'hypervisor_hostname': '2-host'}]
    get_value = {'id': '101', 'hypervisor_hostname': '1-host'}
    show_host, host_manager = self.create_show_command(list_value, get_value)
    args = argparse.Namespace(id='1-host')
    expected = [('hypervisor_hostname', 'id'), ('1-host', '101')]
    ret = show_host.get_data(args)
    self.assertEqual(ret, expected)
    host_manager.get.assert_called_once_with('101')