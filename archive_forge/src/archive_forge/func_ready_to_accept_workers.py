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
def ready_to_accept_workers(self, timeout_seconds=None):
    """
        Set up socket to start communicating to workers.
        """
    assert self.task_state >= self.STATE_INIT_RUN, 'Cannot be ready to accept workers before starting a run with `mturk_manager.start_new_run()` first.'
    shared_utils.print_and_log(logging.INFO, 'Local: Setting up WebSocket...', not self.is_test)
    self._setup_socket(timeout_seconds=timeout_seconds)
    shared_utils.print_and_log(logging.INFO, 'WebSocket set up!', should_print=True)
    if self.STATE_ACCEPTING_WORKERS > self.task_state:
        self.task_state = self.STATE_ACCEPTING_WORKERS