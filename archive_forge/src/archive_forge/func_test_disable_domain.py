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
def test_disable_domain(self):
    domain_ref = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    domain_ref['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain_ref['id'], domain_ref)
    self._assert_notify_sent(domain_ref['id'], 'disabled', 'domain', public=False)