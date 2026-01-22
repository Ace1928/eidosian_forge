from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_list_rsrc_caching(self):
    self.list_rsrcs.return_value = self.resources
    self.insp.size(include_failed=False)
    list(self.insp.member_names(include_failed=True))
    self.insp.size(include_failed=True)
    list(self.insp.member_names(include_failed=False))
    self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))
    self.get_tmpl.assert_not_called()