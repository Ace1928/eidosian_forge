from unittest import mock
from heat.rpc import worker_api as rpc_api
from heat.rpc import worker_client as rpc_client
from heat.tests import common
def test_make_msg(self):
    method = 'sample_method'
    kwargs = {'a': '1', 'b': '2'}
    result = (method, kwargs)
    self.assertEqual(result, rpc_client.WorkerClient.make_msg(method, **kwargs))