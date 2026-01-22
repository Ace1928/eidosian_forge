import datetime
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
from keystone.models import revoke_model
from keystone.tests.unit import test_v3
def test_revoked_list_self_url(self):
    revoked_list_url = '/OS-REVOKE/events'
    resp = self.get(revoked_list_url)
    links = resp.json_body['links']
    self.assertThat(links['self'], matchers.EndsWith(revoked_list_url))