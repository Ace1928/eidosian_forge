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
def test_move_to_waiting(self):
    manager = self.mturk_manager
    manager.socket_manager = mock.MagicMock()
    manager.socket_manager.close_channel = mock.MagicMock()
    manager.force_expire_hit = mock.MagicMock()
    self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
    self.agent_1.reduce_state = mock.MagicMock()
    self.agent_2.reduce_state = mock.MagicMock()
    self.agent_3.reduce_state = mock.MagicMock()
    manager._move_agents_to_waiting([self.agent_1])
    self.agent_1.reduce_state.assert_called_once()
    manager.socket_manager.close_channel.assert_called_once_with(self.agent_1.get_connection_id())
    manager.force_expire_hit.assert_not_called()
    manager.socket_manager.close_channel.reset_mock()
    manager._move_agents_to_waiting([self.agent_2])
    self.agent_2.reduce_state.assert_not_called()
    manager.socket_manager.close_channel.assert_not_called()
    manager.force_expire_hit.assert_not_called()
    manager.accepting_workers = False
    manager._move_agents_to_waiting([self.agent_3])
    self.agent_3.reduce_state.assert_not_called()
    manager.socket_manager.close_channel.assert_not_called()
    manager.force_expire_hit.assert_called_once_with(self.agent_3.worker_id, self.agent_3.assignment_id)