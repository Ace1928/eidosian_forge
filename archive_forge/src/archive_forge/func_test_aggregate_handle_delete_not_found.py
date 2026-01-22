from unittest import mock
from heat.engine.clients.os import nova
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_aggregate_handle_delete_not_found(self):
    ag = mock.MagicMock()
    ag.id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    ag.hosts = ['host_1']
    self.aggregates.get.side_effect = [nova.exceptions.NotFound(404)]
    self.my_aggregate.handle_delete()