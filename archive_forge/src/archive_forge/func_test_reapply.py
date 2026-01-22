import os
import testtools
from unittest import mock
from troveclient.v1 import modules
def test_reapply(self):
    resp = mock.Mock()
    resp.status_code = 200
    body = None
    self.modules.api.client.put = mock.Mock(return_value=(resp, body))
    self.modules.reapply(self.module_name)
    self.modules.reapply(self.module)
    resp.status_code = 500
    self.assertRaises(Exception, self.modules.reapply, self.module_name)