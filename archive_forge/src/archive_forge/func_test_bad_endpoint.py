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
def test_bad_endpoint(self):

    class _endpoint(object):

        def target(self, ctxt, echo):
            return echo
    target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
    transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
    self.assertRaises(TypeError, oslo_messaging.get_rpc_server, transport=transport.transport, target=target, endpoints=[_endpoint()], executor='threading')