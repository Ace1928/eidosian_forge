import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_new_act_message(self):
    """
        Get a new act message if one exists, return None otherwise.
        """
    self.assert_connected()
    if self.msg_queue is not None:
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg['id'] == self.id:
                return msg
    return None