import os
import string
import pytest
from .util import random_string
from keyring import errors
def check_set_get(self, service, username, password):
    keyring = self.keyring
    assert keyring.get_password(service, username) is None
    self.set_password(service, username, password)
    assert keyring.get_password(service, username) == password
    self.set_password(service, username, '')
    assert keyring.get_password(service, username) == ''