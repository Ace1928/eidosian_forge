import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def test_update_host(self):
    list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
    update_host, host_manager = self.create_update_command(list_value)
    args = argparse.Namespace(id='101', extra_capabilities=['key1=value1', 'key2=value2'])
    expected = {'values': {'key1': 'value1', 'key2': 'value2'}}
    update_host.run(args)
    host_manager.update.assert_called_once_with('101', **expected)