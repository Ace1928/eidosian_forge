from unittest import mock
import eventlet
from oslo_context import context
from heat.engine import service
from heat.tests import common
def test_tgm_send(self):
    stack_id = 'send_test'
    e1, e2 = (mock.MagicMock(), mock.Mock())
    thm = service.ThreadGroupManager()
    thm.add_msg_queue(stack_id, e1)
    thm.add_msg_queue(stack_id, e2)
    thm.send(stack_id, 'test_message')