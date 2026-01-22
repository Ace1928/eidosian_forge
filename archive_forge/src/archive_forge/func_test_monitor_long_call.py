import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def test_monitor_long_call(self):
    if not (self.rpc_url.startswith('rabbit://') or self.rpc_url.startswith('amqp://')):
        self.skipTest('backend does not support call monitoring')
    transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
    target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))

    class _endpoint(object):

        def delay(self, ctxt, seconds):
            time.sleep(seconds)
            return seconds
    self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading', endpoint=_endpoint()))
    client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=1)
    self.assertRaises(oslo_messaging.MessagingTimeout, client1.delay, seconds=4)
    client2 = utils.ClientStub(transport.transport, target, cast=False, timeout=3600, call_monitor_timeout=2)
    self.assertEqual(4, client2.delay(seconds=4))