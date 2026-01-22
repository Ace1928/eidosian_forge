import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_delete_lease(self):
    delete_lease, lease_manager = self.create_delete_command()
    lease_manager.delete.return_value = None
    args = argparse.Namespace(id=FIRST_LEASE)
    delete_lease.run(args)
    lease_manager.delete.assert_called_once_with(FIRST_LEASE)