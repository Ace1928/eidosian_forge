from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_no_identity(self):
    self.insp = grouputils.GroupInspector(self.ctx, self.rpc_client, None)
    self.assertEqual(0, self.insp.size(include_failed=True))
    self.assertEqual([], list(self.insp.member_names(include_failed=True)))
    self.assertIsNone(self.insp.template())
    self.list_rsrcs.assert_not_called()
    self.get_tmpl.assert_not_called()