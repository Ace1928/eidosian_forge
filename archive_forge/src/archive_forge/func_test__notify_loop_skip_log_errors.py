from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
@mock.patch('neutron_lib.callbacks.manager.LOG')
def test__notify_loop_skip_log_errors(self, _logger):
    self.manager.subscribe(callback_raise, resources.PORT, events.BEFORE_CREATE)
    self.manager.subscribe(callback_raise, resources.PORT, events.PRECOMMIT_CREATE)
    self.manager._notify_loop(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=mock.ANY)
    self.manager._notify_loop(resources.PORT, events.PRECOMMIT_CREATE, mock.ANY, payload=mock.ANY)
    self.assertFalse(_logger.exception.call_count)
    self.assertTrue(_logger.debug.call_count)