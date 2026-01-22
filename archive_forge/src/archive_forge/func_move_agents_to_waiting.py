import logging
import os
import threading
import time
import uuid
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet
from parlai.mturk.webapp.run_mocks.mock_turk_agent import MockTurkAgent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def move_agents_to_waiting(self, agents):
    """
        Mock moving to a waiting world.
        """
    for agent in agents:
        agent.mock_status = AssignState.STATUS_WAITING
        agent.set_status(AssignState.STATUS_WAITING)
        agent.conversation_id = 'waiting'