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
def test_logging_body_only_for_specified_content_types(self):
    """Verify response body is only logged in specific content types.

        Response bodies are logged only when the response's Content-Type header
        is set to application/json or text/plain. This prevents us to get an
        unexpected MemoryError when reading arbitrary responses, such as
        streams.
        """
    OMITTED_BODY = 'Omitted, Content-Type is set to %s. Only ' + ', '.join(client_session._LOG_CONTENT_TYPES) + ' responses have their bodies logged.'
    session = client_session.Session(verify=False)
    body = json.dumps({'token': {'id': '...'}})
    self.stub_url('POST', text=body)
    session.post(self.TEST_URL)
    self.assertNotIn(body, self.logger.output)
    self.assertIn(OMITTED_BODY % None, self.logger.output)
    body = '<token><id>...</id></token>'
    self.stub_url('POST', text=body, headers={'Content-Type': 'text/xml'})
    session.post(self.TEST_URL)
    self.assertNotIn(body, self.logger.output)
    self.assertIn(OMITTED_BODY % 'text/xml', self.logger.output)
    body = json.dumps({'token': {'id': '...'}})
    self.stub_url('POST', text=body, headers={'Content-Type': 'application/json'})
    session.post(self.TEST_URL)
    self.assertIn(body, self.logger.output)
    self.assertNotIn(OMITTED_BODY % 'application/json', self.logger.output)
    body = json.dumps({'token': {'id': '...'}})
    self.stub_url('POST', text=body, headers={'Content-Type': 'application/json; charset=UTF-8'})
    session.post(self.TEST_URL)
    self.assertIn(body, self.logger.output)
    self.assertNotIn(OMITTED_BODY % 'application/json; charset=UTF-8', self.logger.output)
    text = 'Error detected, unable to continue.'
    self.stub_url('POST', text=text, headers={'Content-Type': 'text/plain'})
    session.post(self.TEST_URL)
    self.assertIn(text, self.logger.output)
    self.assertNotIn(OMITTED_BODY % 'text/plain', self.logger.output)
    text = 'Error detected, unable to continue.'
    self.stub_url('POST', text=text, headers={'Content-Type': 'text/plain; charset=UTF-8'})
    session.post(self.TEST_URL)
    self.assertIn(text, self.logger.output)
    self.assertNotIn(OMITTED_BODY % 'text/plain; charset=UTF-8', self.logger.output)