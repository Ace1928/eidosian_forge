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
def test_bad_request_token_key(self):
    consumer = self._create_single_consumer()
    consumer_id = consumer['id']
    consumer_secret = consumer['secret']
    consumer = {'key': consumer_id, 'secret': consumer_secret}
    url, headers = self._create_request_token(consumer, self.project_id)
    self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
    url = self._authorize_request_token(uuid.uuid4().hex)
    body = {'roles': [{'id': self.role_id}]}
    self.put(url, body=body, expected_status=http.client.NOT_FOUND)