import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_delete_extra_properties(self):
    self.image.extra_properties = {'foo': 'bar'}
    self.assertEqual({'foo': 'bar'}, self.image.extra_properties)
    del self.image.extra_properties['foo']
    self.assertEqual({}, self.image.extra_properties)