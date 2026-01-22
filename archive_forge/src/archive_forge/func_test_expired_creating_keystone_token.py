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
def test_expired_creating_keystone_token(self):
    with freezegun.freeze_time(datetime.datetime.utcnow()) as frozen_time:
        self.config_fixture.config(group='oauth1', access_token_duration=1)
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        self.consumer = {'key': consumer_id, 'secret': consumer_secret}
        self.assertIsNotNone(self.consumer['key'])
        url, headers = self._create_request_token(self.consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        self.request_token = oauth1.Token(request_key, request_secret)
        self.assertIsNotNone(self.request_token.key)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        self.verifier = resp.result['token']['oauth_verifier']
        self.request_token.set_verifier(self.verifier)
        url, headers = self._create_access_token(self.consumer, self.request_token)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        access_key = credentials['oauth_token'][0]
        access_secret = credentials['oauth_token_secret'][0]
        self.access_token = oauth1.Token(access_key, access_secret)
        self.assertIsNotNone(self.access_token.key)
        url, headers, body = self._get_oauth_token(self.consumer, self.access_token)
        frozen_time.tick(delta=datetime.timedelta(seconds=CONF.oauth1.access_token_duration + 1))
        self.post(url, headers=headers, body=body, expected_status=http.client.UNAUTHORIZED)