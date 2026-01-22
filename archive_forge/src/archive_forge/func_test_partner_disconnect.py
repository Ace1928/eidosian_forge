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
def test_partner_disconnect(self):
    manager = self.mturk_manager
    manager.send_state_change = mock.MagicMock()
    self.agent_1.set_status(AssignState.STATUS_IN_TASK)
    manager._handle_partner_disconnect(self.agent_1)
    self.assertEqual(self.agent_1.get_status(), AssignState.STATUS_PARTNER_DISCONNECT)
    args = manager.send_state_change.call_args[0]
    worker_id, assignment_id = (args[0], args[1])
    self.assertEqual(worker_id, self.agent_1.worker_id)
    self.assertEqual(assignment_id, self.agent_1.assignment_id)