from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_service_catalog(self):
    ctx = context.RequestContext(service_catalog=['foo'])
    self.assertEqual(['foo'], ctx.service_catalog)