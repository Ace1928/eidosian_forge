import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_manager_alive_makes_state(self):
    test_worker = self.worker_manager.worker_alive(TEST_WORKER_ID_1)
    self.assertIsInstance(test_worker, WorkerState)
    self.assertEqual(test_worker.worker_id, TEST_WORKER_ID_1)
    self.assertNotEqual(test_worker, self.work_state_1)