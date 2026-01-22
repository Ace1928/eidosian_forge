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
def on_socket_open(*args):
    shared_utils.print_and_log(logging.DEBUG, 'Socket open: {}'.format(args))
    self._send_world_alive()