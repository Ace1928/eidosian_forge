import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_implied_roles_none(self):
    self.prior = self._create_role()
    url = '/roles/%s/implies' % self.prior['id']
    response = self.get(url).json['role_inference']
    self.head(url, expected_status=http.client.OK)
    self.assertEqual(self.prior['id'], response['prior_role']['id'])
    self.assertEqual(0, len(response['implies']))