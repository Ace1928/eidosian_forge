import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def map_over_agents(self, map_function, filter_func=None):
    """
        Take an action over all the agents we have access to, filters if a filter_func
        is given.
        """
    for worker in self.mturk_workers.values():
        for agent in worker.agents.values():
            if filter_func is None or filter_func(agent):
                map_function(agent)