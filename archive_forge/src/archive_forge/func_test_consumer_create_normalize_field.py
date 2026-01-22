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
def test_consumer_create_normalize_field(self):
    field_name = 'some:weird-field'
    field_value = uuid.uuid4().hex
    extra_fields = {field_name: field_value}
    consumer = self._consumer_create(**extra_fields)
    normalized_field_name = 'some_weird_field'
    self.assertEqual(field_value, consumer[normalized_field_name])