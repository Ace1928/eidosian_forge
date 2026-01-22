import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_different_user(self):
    """
        Issue #47 reports that WinVault isn't storing passwords for
        multiple users. This test exercises that test for each of the
        backends.
        """
    keyring = self.keyring
    self.set_password('service1', 'user1', 'password1')
    self.set_password('service1', 'user2', 'password2')
    assert keyring.get_password('service1', 'user1') == 'password1'
    assert keyring.get_password('service1', 'user2') == 'password2'
    self.set_password('service2', 'user3', 'password3')
    assert keyring.get_password('service1', 'user1') == 'password1'