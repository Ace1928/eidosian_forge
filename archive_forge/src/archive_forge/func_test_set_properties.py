import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_set_properties(self, monkeypatch):
    env = dict(KEYRING_PROPERTY_FOO_BAR='fizz buzz', OTHER_SETTING='ignore me')
    monkeypatch.setattr(os, 'environ', env)
    self.keyring.set_properties_from_env()
    assert self.keyring.foo_bar == 'fizz buzz'