import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_difficult_chars(self):
    password = random_string(20, self.DIFFICULT_CHARS)
    username = random_string(20, self.DIFFICULT_CHARS)
    service = random_string(20, self.DIFFICULT_CHARS)
    self.check_set_get(service, username, password)