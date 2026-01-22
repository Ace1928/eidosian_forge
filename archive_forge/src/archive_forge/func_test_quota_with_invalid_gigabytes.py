from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_quota_with_invalid_gigabytes(self):
    fake_v = self.fv(2)
    self.volumes.list.return_value = [fake_v, fake_v]
    self.my_quota.physical_resource_name = mock.MagicMock(return_value='some_resource_id')
    self.my_quota.reparse()
    err = self.assertRaises(ValueError, self.my_quota.handle_create)
    self.assertEqual(self.err_msg % {'property': 'gigabytes', 'value': 5, 'total': 6}, str(err))