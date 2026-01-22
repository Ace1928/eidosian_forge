import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_flavor_delete(self):
    pool_data = {'uri': 'mongodb://127.0.0.1:27017', 'weight': 10, 'flavor': 'tasty'}
    pool = self.client.pool('stomach', **pool_data)
    self.addCleanup(pool.delete)
    pool_list = ['stomach']
    flavor_data = {'pool_list': pool_list}
    flavor = self.client.flavor('tasty', **flavor_data)
    flavor.delete()