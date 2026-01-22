import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_bad_verifier(self):
    self.config_fixture.config(debug=True, insecure_debug=True)
    consumer = self._create_single_consumer()
    consumer_id = consumer['id']
    consumer_secret = consumer['secret']
    consumer = {'key': consumer_id, 'secret': consumer_secret}
    url, headers = self._create_request_token(consumer, self.project_id)
    content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
    credentials = _urllib_parse_qs_text_keys(content.result)
    request_key = credentials['oauth_token'][0]
    request_secret = credentials['oauth_token_secret'][0]
    request_token = oauth1.Token(request_key, request_secret)
    url = self._authorize_request_token(request_key)
    body = {'roles': [{'id': self.role_id}]}
    resp = self.put(url, body=body, expected_status=http.client.OK)
    verifier = resp.result['token']['oauth_verifier']
    self.assertIsNotNone(verifier)
    request_token.set_verifier(uuid.uuid4().hex)
    url, headers = self._create_access_token(consumer, request_token)
    resp = self.post(url, headers=headers, expected_status=http.client.BAD_REQUEST)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Validation failed with errors', resp_data.get('error', {}).get('message'))