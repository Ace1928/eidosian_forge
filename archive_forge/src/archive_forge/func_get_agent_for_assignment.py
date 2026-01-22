import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_agent_for_assignment(self, assignment_id):
    """
        Returns agent for the assignment, none if no agent assigned.
        """
    if assignment_id not in self.assignment_to_worker_id:
        return None
    worker_id = self.assignment_to_worker_id[assignment_id]
    worker = self.mturk_workers[worker_id]
    return worker.get_agent_for_assignment(assignment_id)