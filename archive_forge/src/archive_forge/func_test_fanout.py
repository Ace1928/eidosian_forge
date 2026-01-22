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
def test_fanout(self):
    group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
    client = group.client('all', cast=True)
    client.append(text='open')
    client.append(text='stack')
    client.add(increment=2)
    client.add(increment=10)
    time.sleep(0.3)
    client.sync()
    group.sync(server='all')
    for s in group.servers:
        self.assertIn(s.endpoint.sval, ['openstack', 'stackopen'])
        self.assertEqual(12, s.endpoint.ival)