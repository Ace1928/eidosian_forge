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
def queue_packet(self, packet):
    """
        Queues sending a packet to its intended owner.
        """
    shared_utils.print_and_log(logging.DEBUG, 'Put packet ({}) in queue'.format(packet.id))
    with self.packet_map_lock:
        self.packet_map[packet.id] = packet
    item = (time.time(), packet)
    self.sending_queue.put(item)
    return True