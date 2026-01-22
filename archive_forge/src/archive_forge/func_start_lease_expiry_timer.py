from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
def start_lease_expiry_timer(self, ack_ids: Iterable[str]) -> None:
    """Start the lease expiry timer for `items`.

        Args:
            items: Sequence of ack-ids for which to start lease expiry timers.
        """
    with self._add_remove_lock:
        for ack_id in ack_ids:
            lease_info = self._leased_messages.get(ack_id)
            if lease_info:
                self._leased_messages[ack_id] = lease_info._replace(sent_time=time.time())