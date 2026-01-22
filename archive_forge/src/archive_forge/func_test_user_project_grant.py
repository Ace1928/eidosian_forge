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
def test_user_project_grant(self):
    url = '/projects/%s/users/%s/roles/%s' % (self.project_id, self.user_id, self.role_id)
    self._test_role_assignment(url, self.role_id, project=self.project_id, user=self.user_id)