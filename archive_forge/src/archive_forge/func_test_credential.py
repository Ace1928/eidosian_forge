import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_credential(self):
    keyring = self.keyring
    cred = keyring.get_credential('service', None)
    assert cred is None
    self.set_password('service1', 'user1', 'password1')
    self.set_password('service1', 'user2', 'password2')
    cred = keyring.get_credential('service1', None)
    assert cred is None or (cred.username, cred.password) in (('user1', 'password1'), ('user2', 'password2'))
    cred = keyring.get_credential('service1', 'user2')
    assert cred is not None
    assert (cred.username, cred.password) in (('user1', 'password1'), ('user2', 'password2'))