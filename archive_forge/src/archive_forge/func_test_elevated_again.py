from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_elevated_again(self):
    """Make sure a second elevation looks the same."""
    ctx = context.RequestContext(service_catalog=['foo'], user_id='dan', project_id='openstack', roles=['member'])
    admin = ctx.elevated()
    admin = admin.elevated()
    self.assertEqual('dan', admin.user_id)
    self.assertEqual('openstack', admin.project_id)
    self.assertEqual(sorted(['member', 'admin']), sorted(admin.roles))
    self.assertEqual(['foo'], admin.service_catalog)
    self.assertTrue(admin.is_admin)