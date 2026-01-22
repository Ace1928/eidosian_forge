import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def register_to_conv(self, agent, conversation_id):
    """
        Handle registering an agent to a particular conversation.

        Should be called by an agent whenever given a conversation id
        """
    if conversation_id not in self.conv_to_agent:
        self.conv_to_agent[conversation_id] = []
    self.conv_to_agent[conversation_id].append(agent)