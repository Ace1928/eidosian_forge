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
def test_share_check(self):
    share = self._create_share('stack_share_check')
    scheduler.TaskRunner(share.check)()
    expected_state = (share.CHECK, share.COMPLETE)
    self.assertEqual(expected_state, share.state, 'Share is not in expected state')