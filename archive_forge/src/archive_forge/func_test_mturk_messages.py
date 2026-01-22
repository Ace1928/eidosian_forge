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
def test_mturk_messages(self):
    """
        Ensure incoming messages work as expected.
        """
    manager = self.mturk_manager
    manager.task_group_id = 'TEST_GROUP_ID'
    manager.server_url = 'https://127.0.0.1'
    manager.task_state = manager.STATE_ACCEPTING_WORKERS
    manager._setup_socket()
    manager.force_expire_hit = mock.MagicMock()
    manager._on_socket_dead = mock.MagicMock()
    alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
    manager._on_alive(alive_packet)
    agent = manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
    self.assertIn(agent.get_status(), [AssignState.STATUS_NONE, AssignState.STATUS_WAITING])
    self.assertIsInstance(agent, MTurkAgent)
    manager._on_socket_dead = mock.MagicMock()
    message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_ABANDONDED}, '')
    manager._handle_mturk_message(message_packet)
    manager._on_socket_dead.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    manager._on_socket_dead.reset_mock()
    message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_RETURNED}, '')
    agent.hit_is_returned = False
    manager._handle_mturk_message(message_packet)
    manager._on_socket_dead.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    manager._on_socket_dead.reset_mock()
    self.assertTrue(agent.hit_is_returned)
    message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED}, '')
    agent.hit_is_complete = False
    manager._handle_mturk_message(message_packet)
    manager._on_socket_dead.assert_not_called()
    self.assertTrue(agent.hit_is_complete)