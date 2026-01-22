from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
def test_get_client(self):
    client_manager = self.app.client_manager
    del self.app.client_manager
    client = self.command.get_client()
    self.assertEqual(self.app.client, client)
    self.app.client_manager = client_manager
    del self.app.client
    client = self.command.get_client()
    self.assertEqual(self.app.client_manager.reservation, client)