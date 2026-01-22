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
def test_get_role_assignments_by_effective_and_inherited(self):
    self.get_role_assignments(domain_id=self.domain_id, effective=True, inherited_to_projects=True, expected_status=http.client.BAD_REQUEST)