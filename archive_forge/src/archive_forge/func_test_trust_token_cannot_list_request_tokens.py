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
def test_trust_token_cannot_list_request_tokens(self):
    self._set_policy({'identity:list_access_tokens': [], 'identity:create_trust': []})
    trust_token = self._create_trust_get_token()
    url = '/users/%s/OS-OAUTH1/access_tokens' % self.user_id
    self.get(url, token=trust_token, expected_status=http.client.FORBIDDEN)