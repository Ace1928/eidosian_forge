import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def un_time_block_workers(self, workers=None):
    if workers is None:
        workers = self.time_blocked_workers
        self.time_blocked_workers = []
    for worker_id in workers:
        self.mturk_manager.un_soft_block_worker(worker_id, 'max_time_qual')