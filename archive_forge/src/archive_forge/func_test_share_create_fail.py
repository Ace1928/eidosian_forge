import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_share_create_fail(self):
    share = self._init_share('stack_share_create_fail')
    share.client().shares.get.return_value = self.failed_share
    exc = self.assertRaises(exception.ResourceInError, share.check_create_complete, self.failed_share)
    self.assertIn('Error during creation', str(exc))