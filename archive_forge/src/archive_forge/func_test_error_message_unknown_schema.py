import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_error_message_unknown_schema(self):
    error_message = 'Uh oh, things went bad!'
    payload = json.dumps(error_message)
    self.stub_url('GET', status_code=9000, text=payload, headers={'Content-Type': 'application/json'})
    session = client_session.Session()
    msg = 'Unrecognized schema in response body. (HTTP 9000)'
    try:
        session.get(self.TEST_URL)
    except exceptions.HttpError as ex:
        self.assertEqual(ex.message, msg)