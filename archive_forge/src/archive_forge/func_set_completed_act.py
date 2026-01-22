import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def set_completed_act(self, completed_act):
    """
        Set the completed act for an agent, notes successful submission.
        """
    self.completed_act = completed_act
    self.hit_is_complete = True