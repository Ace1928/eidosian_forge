import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def assign_task_to_worker(self, hit_id, assign_id, worker_id):
    self.assignment_to_worker_id[assign_id] = worker_id
    agent = self._create_agent(hit_id, assign_id, worker_id)
    self.hit_id_to_agent[hit_id] = agent
    curr_worker_state = self.mturk_workers[worker_id]
    curr_worker_state.add_agent(agent)