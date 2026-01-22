from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def get_member_assignments():
    assignments = PROVIDERS.assignment_api.list_role_assignments()
    return [x for x in assignments if x['role_id'] == default_fixtures.MEMBER_ROLE_ID]