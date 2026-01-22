import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_pool_create(self):
    pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
    pool = self.client.pool('FuncTestPool', **pool_data)
    self.addCleanup(pool.delete)
    self.assertEqual('FuncTestPool', pool.name)
    self.assertEqual(10, pool.weight)