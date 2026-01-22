import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def update_agent_id(self, agent_id):
    """
        Workaround used to force an update to an agent_id on the front-end to render the
        correct react components for onboarding and waiting worlds.

        Only really used in special circumstances where different agents need different
        onboarding worlds.
        """
    update_packet = {'agent_id': agent_id}
    self.m_send_state_change(self.worker_id, self.assignment_id, update_packet)