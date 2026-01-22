import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_set_after_set_blank(self):
    service = random_string(20)
    username = random_string(20)
    self.keyring.set_password(service, username, '')
    self.keyring.set_password(service, username, 'non-blank')