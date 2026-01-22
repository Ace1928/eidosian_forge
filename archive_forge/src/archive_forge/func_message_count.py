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
@property
def message_count(self) -> int:
    """The number of leased messages."""
    return len(self._leased_messages)