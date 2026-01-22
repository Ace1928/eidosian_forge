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
def test_validate_access_token_request_failed(self):
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
    request_token.set_verifier(verifier)
    base_url = 'http://localhost/identity_admin/v3'
    url, headers = self._create_access_token(consumer, request_token, base_url=base_url)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
    bad_url_scheme = self._switch_baseurl_scheme()
    url, headers = self._create_access_token(consumer, request_token, base_url=bad_url_scheme)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
    consumer.update({'secret': uuid.uuid4().hex})
    url, headers = self._create_access_token(consumer, request_token)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
    verifier = ''.join(random.SystemRandom().sample(base.VERIFIER_CHARS, 8))
    request_token.set_verifier(verifier)
    url, headers = self._create_access_token(consumer, request_token)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Provided verifier', resp_data.get('error', {}).get('message'))
    consumer.update({'key': uuid.uuid4().hex})
    url, headers = self._create_access_token(consumer, request_token)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Provided consumer does not exist', resp_data.get('error', {}).get('message'))
    consumer2 = self._create_single_consumer()
    consumer.update({'key': consumer2['id']})
    url, headers = self._create_access_token(consumer, request_token)
    resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Provided consumer key', resp_data.get('error', {}).get('message'))