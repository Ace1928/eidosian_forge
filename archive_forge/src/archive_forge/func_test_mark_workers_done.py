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
def test_mark_workers_done(self):
    manager = self.mturk_manager
    manager.send_state_change = mock.MagicMock()
    manager.give_worker_qualification = mock.MagicMock()
    manager._log_working_time = mock.MagicMock()
    manager.has_time_limit = False
    self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
    manager.mark_workers_done([self.agent_1])
    self.assertEqual(AssignState.STATUS_DISCONNECT, self.agent_1.get_status())
    manager.is_unique = True
    with self.assertRaises(AssertionError):
        manager.mark_workers_done([self.agent_2])
    manager.give_worker_qualification.assert_not_called()
    manager.unique_qual_name = 'fake_qual_name'
    manager.mark_workers_done([self.agent_2])
    manager.give_worker_qualification.assert_called_once_with(self.agent_2.worker_id, 'fake_qual_name')
    self.assertEqual(self.agent_2.get_status(), AssignState.STATUS_DONE)
    manager.is_unique = False
    manager.has_time_limit = True
    manager.mark_workers_done([self.agent_3])
    self.assertEqual(self.agent_3.get_status(), AssignState.STATUS_DONE)
    manager._log_working_time.assert_called_once_with(self.agent_3)