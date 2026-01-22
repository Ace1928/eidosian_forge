from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_deserialize_context_no_ids(self):
    context_dict = {'foo': 'bar', 'is_admin': True}
    c = self.ser.deserialize_context(context_dict)
    self.assertIsNone(c.user_id)
    self.assertIsNone(c.project_id)