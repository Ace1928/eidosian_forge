import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def expire_all_unassigned_hits(self):
    """
        Move through the whole unassigned hit list and attempt to expire the HITs,
        though this only immediately expires those that aren't assigned.
        """
    shared_utils.print_and_log(logging.INFO, 'Expiring all unassigned HITs...', should_print=not self.is_test)
    for hit_id in self.hit_id_list:
        mturk_utils.expire_hit(self.is_sandbox, hit_id)