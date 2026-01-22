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
def test_event_registration_for_unknown_resource_type(self):
    manager = self.FakeManager()
    notifications.register_event_callback(DELETED_OPERATION, uuid.uuid4().hex, manager._project_deleted_callback)
    resource_type = uuid.uuid4().hex
    notifications.register_event_callback(DELETED_OPERATION, resource_type, manager._project_deleted_callback)