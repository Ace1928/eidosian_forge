import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet, SocketManager, StaticSocketManager
from parlai.mturk.core.worker_manager import WorkerManager
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.server_utils as server_utils
import parlai.mturk.core.shared_utils as shared_utils
def send_state_data():
    while not agent.alived and (not agent.hit_is_expired):
        time.sleep(shared_utils.THREAD_SHORT_SLEEP)
    data = {'text': data_model.COMMAND_RESTORE_STATE, 'messages': agent.get_messages(), 'last_command': agent.get_last_command()}
    self.send_command(worker_id, assignment_id, data)
    if agent.message_request_time is not None:
        agent.request_message()