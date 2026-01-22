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
def test_deleting_project_also_invalidates_tokens(self):
    self.test_oauth_flow()
    headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
    r = self.get('/auth/tokens', headers=headers)
    self.assertValidTokenResponse(r, self.user)
    r = self.delete('/projects/%(project_id)s' % {'project_id': self.project_id})
    headers = {'X-Subject-Token': self.keystone_token_id}
    self.get(path='/auth/tokens', token=self.get_admin_token(), headers=headers, expected_status=http.client.NOT_FOUND)