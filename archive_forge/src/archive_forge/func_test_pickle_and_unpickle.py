import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
def test_pickle_and_unpickle(self):
    creds = self.make_credentials()
    unpickled = pickle.loads(pickle.dumps(creds))
    assert list(creds.__dict__).sort() == list(unpickled.__dict__).sort()
    for attr in list(creds.__dict__):
        assert getattr(creds, attr) == getattr(unpickled, attr)