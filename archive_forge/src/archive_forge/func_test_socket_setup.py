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
def test_socket_setup(self):
    """
        Basic socket setup should fail when not in correct state, but succeed otherwise.
        """
    self.mturk_manager.task_state = self.mturk_manager.STATE_CREATED
    with self.assertRaises(AssertionError):
        self.mturk_manager._setup_socket()
    self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
    self.mturk_manager.server_url = 'https://127.0.0.1'
    self.mturk_manager.task_state = self.mturk_manager.STATE_INIT_RUN
    self.mturk_manager._setup_socket()
    self.assertIsInstance(self.mturk_manager.socket_manager, SocketManager)