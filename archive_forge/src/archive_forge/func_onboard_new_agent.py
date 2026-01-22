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
def onboard_new_agent(self, agent):
    """
        Creates an onboarding thread for the given agent.
        """
    worker_id = agent.worker_id
    assignment_id = agent.assignment_id

    def _onboard_function(agent):
        """
            Onboarding wrapper to set state to onboarding properly.
            """
        if self.onboard_function:
            agent.id = 'Onboarding'
            self.onboard_function(agent)
        self.move_agents_to_waiting([agent])
    onboard_thread = threading.Thread(target=_onboard_function, args=(agent,), name='onboard-{}-{}'.format(worker_id, assignment_id))
    onboard_thread.daemon = True
    onboard_thread.start()
    return True