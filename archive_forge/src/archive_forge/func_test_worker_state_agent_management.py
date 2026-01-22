import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_worker_state_agent_management(self):
    """
        Test public state management methods of worker_state.
        """
    agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
    agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_1)
    agent_3 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_3, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_3)
    self.assertEqual(self.work_state_1.active_conversation_count(), 0)
    self.work_state_1.add_agent(agent_1)
    self.assertEqual(self.work_state_1.active_conversation_count(), 1)
    self.work_state_1.add_agent(agent_2)
    self.assertEqual(self.work_state_1.active_conversation_count(), 2)
    with self.assertRaises(AssertionError):
        self.work_state_1.add_agent(agent_3)
    self.assertEqual(self.work_state_1.active_conversation_count(), 2)
    self.assertEqual(self.work_state_1.completed_assignments(), 0)
    self.assertTrue(self.work_state_1.has_assignment(agent_1.assignment_id))
    self.assertTrue(self.work_state_1.has_assignment(agent_2.assignment_id))
    self.assertFalse(self.work_state_1.has_assignment(agent_3.assignment_id))
    self.assertEqual(agent_1, self.work_state_1.get_agent_for_assignment(agent_1.assignment_id))
    self.assertEqual(agent_2, self.work_state_1.get_agent_for_assignment(agent_2.assignment_id))
    self.assertIsNone(self.work_state_1.get_agent_for_assignment(agent_3.assignment_id))
    agent_1.set_status(AssignState.STATUS_DONE)
    self.assertEqual(self.work_state_1.active_conversation_count(), 1)
    self.assertEqual(self.work_state_1.completed_assignments(), 1)
    agent_2.set_status(AssignState.STATUS_DISCONNECT)
    self.assertEqual(self.work_state_1.active_conversation_count(), 0)
    self.assertEqual(self.work_state_1.completed_assignments(), 1)