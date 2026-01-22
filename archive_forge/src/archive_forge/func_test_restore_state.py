import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.worker_manager import WorkerManager
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.core.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model
def test_restore_state(self):
    manager = self.mturk_manager
    worker_manager = manager.worker_manager
    worker_manager.change_agent_conversation = mock.MagicMock()
    manager.send_command = mock.MagicMock()
    agent = self.agent_1
    agent.conversation_id = 'Test_conv_id'
    agent.id = 'test_agent_id'
    agent.request_message = mock.MagicMock()
    agent.message_request_time = time.time()
    test_message = {'text': 'this_is_a_message', 'message_id': 'test_id', 'type': data_model.MESSAGE_TYPE_MESSAGE}
    agent.append_message(test_message)
    manager._restore_agent_state(agent.worker_id, agent.assignment_id)
    self.assertFalse(agent.alived)
    manager.send_command.assert_not_called()
    worker_manager.change_agent_conversation.assert_called_once_with(agent=agent, conversation_id=agent.conversation_id, new_agent_id=agent.id)
    agent.alived = True
    assert_equal_by(lambda: len(agent.request_message.call_args_list), 1, 0.6)
    manager.send_command.assert_called_once()
    args = manager.send_command.call_args[0]
    worker_id, assignment_id, data = (args[0], args[1], args[2])
    self.assertEqual(worker_id, agent.worker_id)
    self.assertEqual(assignment_id, agent.assignment_id)
    self.assertListEqual(data['messages'], agent.get_messages())
    self.assertEqual(data['text'], data_model.COMMAND_RESTORE_STATE)