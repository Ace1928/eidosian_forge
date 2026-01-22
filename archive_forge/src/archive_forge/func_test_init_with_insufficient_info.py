from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
def test_init_with_insufficient_info(self):
    self.assertRaises(exception.InsufficientAuthInformation, base.BaseClientManager, blazar_url=None, auth_token=self.auth_token, session=None)