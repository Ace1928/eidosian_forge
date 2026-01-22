import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_private_create_agent(self):
    """
        Check create agent method used internally in worker_manager.
        """
    test_agent = self.worker_manager._create_agent(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
    self.assertIsInstance(test_agent, MTurkAgent)
    self.assertEqual(test_agent.worker_id, TEST_WORKER_ID_1)
    self.assertEqual(test_agent.hit_id, TEST_HIT_ID_1)
    self.assertEqual(test_agent.assignment_id, TEST_ASSIGNMENT_ID_1)