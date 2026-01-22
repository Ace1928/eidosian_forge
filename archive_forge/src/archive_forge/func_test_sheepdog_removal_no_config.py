import glance_store
from oslo_config import cfg
from oslo_upgradecheck import upgradecheck
from glance.cmd.status import Checks
from glance.tests import utils as test_utils
def test_sheepdog_removal_no_config(self):
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)