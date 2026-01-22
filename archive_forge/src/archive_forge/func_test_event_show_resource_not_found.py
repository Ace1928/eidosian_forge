import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
def test_event_show_resource_not_found(self):
    error = 'Resource not found'
    self.stack_client.get.side_effect = exc.HTTPNotFound(error)
    self._test_not_found(error)