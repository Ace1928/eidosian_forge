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
def test_server_in_group(self):
    if self.rpc_url.startswith('amqp:'):
        self.skipTest('QPID-6307')
    group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
    client = group.client(cast=True)
    for i in range(20):
        client.add(increment=1)
    for i in range(len(group.servers)):
        client.sync()
    group.sync(server='all')
    total = 0
    for s in group.servers:
        ival = s.endpoint.ival
        self.assertThat(ival, matchers.GreaterThan(0))
        self.assertThat(ival, matchers.LessThan(20))
        total += ival
    self.assertEqual(20, total)