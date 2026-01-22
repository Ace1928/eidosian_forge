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
def test_qualification_management(self):
    manager = self.mturk_manager
    test_qual_name = 'test_qual'
    other_qual_name = 'other_qual'
    test_qual_id = 'test_qual_id'
    worker_id = self.agent_1.worker_id
    mturk_utils = MTurkManagerFile.mturk_utils
    success_id = 'Success'

    def find_qualification(qual_name, _sandbox):
        if qual_name == test_qual_name:
            return test_qual_id
        return None
    mturk_utils.find_qualification = find_qualification
    mturk_utils.give_worker_qualification = mock.MagicMock()
    mturk_utils.remove_worker_qualification = mock.MagicMock()
    mturk_utils.find_or_create_qualification = mock.MagicMock(return_value=success_id)
    manager.give_worker_qualification(worker_id, test_qual_name)
    mturk_utils.give_worker_qualification.assert_called_once_with(worker_id, test_qual_id, None, manager.is_sandbox)
    manager.remove_worker_qualification(worker_id, test_qual_name)
    mturk_utils.remove_worker_qualification.assert_called_once_with(worker_id, test_qual_id, manager.is_sandbox, '')
    result = manager.create_qualification(test_qual_name, '')
    self.assertEqual(result, success_id)
    result = manager.create_qualification(test_qual_name, '', False)
    self.assertIsNone(result)
    result = manager.create_qualification(other_qual_name, '')
    self.assertEqual(result, success_id)