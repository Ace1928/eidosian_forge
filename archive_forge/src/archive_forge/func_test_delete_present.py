import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_delete_present(self):
    password = random_string(20, self.DIFFICULT_CHARS)
    username = random_string(20, self.DIFFICULT_CHARS)
    service = random_string(20, self.DIFFICULT_CHARS)
    self.keyring.set_password(service, username, password)
    self.keyring.delete_password(service, username)
    assert self.keyring.get_password(service, username) is None