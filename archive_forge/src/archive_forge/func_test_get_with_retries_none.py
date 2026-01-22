import re
from unittest import mock
import ddt
import requests
import manilaclient
from manilaclient.common import httpclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_get_with_retries_none(self):
    cl = get_authed_client(retries=None)

    @mock.patch.object(requests, 'request', bad_401_request)
    def test_get_call():
        resp, body = cl.get('/hi')
    self.assertRaises(exceptions.Unauthorized, test_get_call)