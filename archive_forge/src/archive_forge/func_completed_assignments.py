import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def completed_assignments(self):
    """
        Returns the number of assignments this worker has completed.
        """
    complete_count = 0
    for agent in self.agents.values():
        if agent.get_status() == AssignState.STATUS_DONE:
            complete_count += 1
    return complete_count