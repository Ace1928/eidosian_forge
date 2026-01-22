import glance_store
from oslo_config import cfg
from oslo_upgradecheck import upgradecheck
from glance.cmd.status import Checks
from glance.tests import utils as test_utils
def test_sheepdog_removal_glance_store_stores(self):
    self.config(stores=None, group='glance_store')
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(stores='', group='glance_store')
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(stores='foo', group='glance_store')
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(stores='sheepdog', group='glance_store')
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.FAILURE)