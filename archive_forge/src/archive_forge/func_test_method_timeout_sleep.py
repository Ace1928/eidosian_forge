from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_method_timeout_sleep(self):
    rpc.TRANSPORT.conf.rpc_response_timeout = 2
    for i in range(100):
        with testtools.ExpectedException(messaging.MessagingTimeout):
            self.client.call(self.call_context, 'method_1')
        self.assertGreaterEqual(self.sleep.call_args_list[0][0][0], 0)
        self.assertLessEqual(self.sleep.call_args_list[0][0][0], 2)
        self.sleep.reset_mock()