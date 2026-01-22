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
def get_agent_work_status(self, assignment_id):
    return self.worker_manager.get_agent_work_status(assignment_id)