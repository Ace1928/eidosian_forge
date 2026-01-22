from unittest import mock
from heat.rpc import worker_api as rpc_api
from heat.rpc import worker_client as rpc_client
from heat.tests import common
def test_cancel_check_resource(self):
    mock_stack_id = 'dummy-stack-id'
    mock_cnxt = mock.Mock()
    method = 'cancel_check_resource'
    kwargs = {'stack_id': mock_stack_id}
    mock_rpc_client = mock.MagicMock()
    mock_cast = mock.MagicMock()
    with mock.patch('heat.common.messaging.get_rpc_client') as mock_grc:
        mock_grc.return_value = mock_rpc_client
        mock_rpc_client.prepare.return_value = mock_cast
        wc = rpc_client.WorkerClient()
        ret_val = wc.cancel_check_resource(mock_cnxt, mock_stack_id, self.fake_engine_id)
        mock_grc.assert_called_with(version=wc.BASE_RPC_API_VERSION, topic=rpc_api.TOPIC, server=self.fake_engine_id)
        self.assertIsNone(ret_val)
        mock_rpc_client.prepare.assert_called_with(version='1.3')
        mock_cast.cast.assert_called_with(mock_cnxt, method, **kwargs)