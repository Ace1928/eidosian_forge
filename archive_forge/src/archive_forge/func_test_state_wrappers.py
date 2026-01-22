import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_state_wrappers(self):
    """
        Test the mturk agent wrappers around its state.
        """
    for status in statuses:
        self.turk_agent.set_status(status)
        self.assertEqual(self.turk_agent.get_status(), status)
    for status in [AssignState.STATUS_DONE, AssignState.STATUS_PARTNER_DISCONNECT]:
        self.turk_agent.set_status(status)
        self.assertTrue(self.turk_agent.submitted_hit())
    for status in active_statuses:
        self.turk_agent.set_status(status)
        self.assertFalse(self.turk_agent.is_final())
    for status in complete_statuses:
        self.turk_agent.set_status(status)
        self.assertTrue(self.turk_agent.is_final())
    self.turk_agent.state.append_message(MESSAGE_1)
    self.assertEqual(len(self.turk_agent.get_messages()), 1)
    self.turk_agent.state.append_message(MESSAGE_2)
    self.assertEqual(len(self.turk_agent.get_messages()), 2)
    self.turk_agent.state.append_message(MESSAGE_1)
    self.assertEqual(len(self.turk_agent.get_messages()), 2)
    self.assertIn(MESSAGE_1, self.turk_agent.get_messages())
    self.assertIn(MESSAGE_2, self.turk_agent.get_messages())
    self.turk_agent.clear_messages()
    self.assertEqual(len(self.turk_agent.get_messages()), 0)