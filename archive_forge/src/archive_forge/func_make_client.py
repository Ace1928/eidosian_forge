import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
@classmethod
def make_client(cls, client_auth=None):
    return sts.Client(cls.TOKEN_EXCHANGE_ENDPOINT, client_auth)