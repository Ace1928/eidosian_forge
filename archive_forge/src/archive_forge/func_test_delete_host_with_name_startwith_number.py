import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def test_delete_host_with_name_startwith_number(self):
    list_value = [{'id': '101', 'hypervisor_hostname': '1-host'}, {'id': '201', 'hypervisor_hostname': '2-host'}]
    delete_host, host_manager = self.create_delete_command(list_value)
    args = argparse.Namespace(id='1-host')
    delete_host.run(args)
    host_manager.delete.assert_called_once_with('101')