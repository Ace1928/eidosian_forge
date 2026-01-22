import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_worker_state_init(self):
    """
        Test proper initialization of worker states.
        """
    self.assertEqual(self.work_state_1.worker_id, TEST_WORKER_ID_1)
    self.assertEqual(self.work_state_2.worker_id, TEST_WORKER_ID_2)
    self.assertEqual(self.work_state_1.disconnects, 10)
    self.assertEqual(self.work_state_2.disconnects, 0)