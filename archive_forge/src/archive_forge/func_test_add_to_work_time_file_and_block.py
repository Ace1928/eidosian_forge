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
def test_add_to_work_time_file_and_block(self):
    manager = self.mturk_manager
    self.agent_1.creation_time = 1000
    self.agent_2.creation_time = 1000
    manager.opt['max_time'] = 10000
    MTurkManagerFile.time.time = mock.MagicMock(return_value=10000)
    self.mturk_manager._should_use_time_logs = mock.MagicMock(return_value=True)
    manager._log_working_time(self.agent_1)
    manager.worker_manager.time_block_worker.assert_not_called()
    MTurkManagerFile.time.time = mock.MagicMock(return_value=100000)
    manager._log_working_time(self.agent_2)
    manager.worker_manager.time_block_worker.assert_called_with(self.agent_2.worker_id)
    manager._reset_time_logs(force=True)
    manager.worker_manager.un_time_block_workers.assert_called_once()
    args = manager.worker_manager.un_time_block_workers.call_args
    worker_list = args[0][0]
    self.assertIn(self.agent_1.worker_id, worker_list)
    self.assertIn(self.agent_2.worker_id, worker_list)