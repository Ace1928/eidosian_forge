import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_force_expire_hit(self):
    manager = self.mturk_manager
    agent = self.agent_1
    worker_id = agent.worker_id
    assignment_id = agent.assignment_id
    socket_manager = manager.socket_manager
    manager.send_command = mock.MagicMock()
    manager.send_state_change = mock.MagicMock()
    socket_manager.close_channel = mock.MagicMock()
    agent.set_status(AssignState.STATUS_DONE)
    manager.force_expire_hit(worker_id, assignment_id)
    manager.send_command.assert_not_called()
    socket_manager.close_channel.assert_not_called()
    self.assertEqual(agent.get_status(), AssignState.STATUS_DONE)
    agent.set_status(AssignState.STATUS_ONBOARDING)
    manager.send_state_change.reset_mock()
    manager.force_expire_hit(worker_id, assignment_id)
    manager.send_state_change.assert_called_once()
    args = manager.send_state_change.call_args[0]
    used_worker_id, used_assignment_id, data = (args[0], args[1], args[2])
    ack_func = manager.send_state_change.call_args[1]['ack_func']
    ack_func()
    self.assertEqual(worker_id, used_worker_id)
    self.assertEqual(assignment_id, used_assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
    self.assertTrue(agent.hit_is_expired)
    self.assertIsNotNone(data['done_text'])
    socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    agent.set_status(AssignState.STATUS_ONBOARDING)
    agent.hit_is_expired = False
    manager.send_state_change.reset_mock()
    socket_manager.close_channel = mock.MagicMock()
    special_disconnect_text = 'You were disconnected as part of a test'
    test_ack_function = mock.MagicMock()
    manager.force_expire_hit(worker_id, assignment_id, text=special_disconnect_text, ack_func=test_ack_function)
    manager.send_state_change.assert_called_once()
    args = manager.send_state_change.call_args[0]
    used_worker_id, used_assignment_id, data = (args[0], args[1], args[2])
    ack_func = manager.send_state_change.call_args[1]['ack_func']
    ack_func()
    self.assertEqual(worker_id, used_worker_id)
    self.assertEqual(assignment_id, used_assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
    self.assertTrue(agent.hit_is_expired)
    self.assertEqual(data['done_text'], special_disconnect_text)
    socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    test_ack_function.assert_called()