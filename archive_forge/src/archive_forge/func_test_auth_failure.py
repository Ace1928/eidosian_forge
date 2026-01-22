import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_auth_failure(self):
    cl = get_client()

    @mock.patch.object(requests, 'request', mock_request)
    def test_auth_call():
        self.assertRaises(exceptions.AuthorizationFailure, cl.authenticate)
    test_auth_call()