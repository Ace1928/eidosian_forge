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
def get_my_sender_id(self):
    """
        Gives the name that this socket manager should use for its world.
        """
    return '[World_{}]'.format(self.task_group_id)