import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_connection_id(self):
    """
        Returns an appropriate connection_id for this agent.
        """
    return '{}_{}'.format(self.worker_id, self.assignment_id)