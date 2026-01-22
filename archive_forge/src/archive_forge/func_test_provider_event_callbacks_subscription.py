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
def test_provider_event_callbacks_subscription(self):
    callback_called = []

    @notifications.listener
    class Foo(object):

        def __init__(self):
            self.event_callbacks = {CREATED_OPERATION: {'project': [self.callback_0, self.callback_1]}}

        def callback_0(self, service, resource_type, operation, payload):
            callback_called.append('cb0')

        def callback_1(self, service, resource_type, operation, payload):
            callback_called.append('cb1')
    Foo()
    project_ref = unit.new_project_ref(domain_id=self.domain_id)
    PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    self.assertCountEqual(['cb1', 'cb0'], callback_called)