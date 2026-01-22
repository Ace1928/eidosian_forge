import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_status_change(self):
    self.turk_agent.set_status(AssignState.STATUS_ONBOARDING)
    time.sleep(0.07)
    self.assertEqual(self.turk_agent.get_status(), AssignState.STATUS_ONBOARDING)
    self.turk_agent.set_status(AssignState.STATUS_WAITING)
    time.sleep(0.07)
    self.assertEqual(self.turk_agent.get_status(), AssignState.STATUS_WAITING)