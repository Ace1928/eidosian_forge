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
def test_deleting_consumer_also_deletes_tokens(self):
    self.test_oauth_flow()
    consumer_id = self.consumer['key']
    resp = self.delete('/OS-OAUTH1/consumers/%(consumer_id)s' % {'consumer_id': consumer_id})
    self.assertResponseStatus(resp, http.client.NO_CONTENT)
    resp = self.get('/users/%(user_id)s/OS-OAUTH1/access_tokens' % {'user_id': self.user_id})
    entities = resp.result['access_tokens']
    self.assertEqual([], entities)
    headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
    self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)