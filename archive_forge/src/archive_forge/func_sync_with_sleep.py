from unittest import mock
from oslo_db import exception
from heat.engine import sync_point
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch('heat.engine.sync_point.update_input_data', return_value=None)
@mock.patch('time.sleep', side_effect=exception.DBError)
def sync_with_sleep(self, ctx, stack, mock_sleep_time, mock_uid):
    resource = stack['C']
    graph = stack.convergence_dependencies.graph()
    mock_callback = mock.Mock()
    sender = (3, True)
    self.assertRaises(exception.DBError, sync_point.sync, ctx, resource.id, stack.current_traversal, True, mock_callback, set(graph[resource.id, True]), {sender: None})
    return mock_sleep_time