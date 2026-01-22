import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def test_delete_host(self):
    list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
    delete_host, host_manager = self.create_delete_command(list_value)
    args = argparse.Namespace(id='101')
    delete_host.run(args)
    host_manager.delete.assert_called_once_with('101')