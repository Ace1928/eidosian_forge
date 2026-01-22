import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_role(self):
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    PROVIDERS.role_api.update_role(role_ref['id'], role_ref)
    self._assert_last_note(role_ref['id'], UPDATED_OPERATION, 'role')
    self._assert_last_audit(role_ref['id'], UPDATED_OPERATION, 'role', cadftaxonomy.SECURITY_ROLE)