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
def test_timeout_with_concurrently_queues(self):
    transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
    target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
    server = self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading'))
    client = utils.ClientStub(transport.transport, target, cast=False, timeout=5)

    def short_periodical_tasks():
        for i in range(10):
            client.add(increment=1)
            time.sleep(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(client.long_running_task, seconds=10)
        executor.submit(short_periodical_tasks)
        self.assertRaises(oslo_messaging.MessagingTimeout, future.result)
    self.assertEqual(10, server.endpoint.ival)