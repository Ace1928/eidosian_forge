import json
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
@mock.patch.object(cs.client, 'authenticate')
def test_auth_call(m):
    cs.authenticate()
    self.assertTrue(m.called)