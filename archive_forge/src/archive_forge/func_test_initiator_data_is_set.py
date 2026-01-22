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
def test_initiator_data_is_set(self):
    ref = unit.new_domain_ref()
    resp = self.post('/domains', body={'domain': ref})
    resource_id = resp.result.get('domain').get('id')
    self._assert_last_audit(resource_id, CREATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)
    self._assert_initiator_data_is_set(CREATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)