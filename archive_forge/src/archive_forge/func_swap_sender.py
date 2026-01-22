import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.dev.shared_utils import print_and_log
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def swap_sender(self):
    """
        Swaps the sender_id and receiver_id.
        """
    self.sender_id, self.receiver_id = (self.receiver_id, self.sender_id)
    return self