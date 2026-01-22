import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def test_show_host(self):
    list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
    get_value = {'id': '101', 'hypervisor_hostname': 'host-1'}
    show_host, host_manager = self.create_show_command(list_value, get_value)
    args = argparse.Namespace(id='101')
    expected = [('hypervisor_hostname', 'id'), ('host-1', '101')]
    ret = show_host.get_data(args)
    self.assertEqual(ret, expected)
    host_manager.get.assert_called_once_with('101')