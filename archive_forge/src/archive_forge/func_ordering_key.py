from __future__ import absolute_import
import datetime as dt
import json
import math
import time
import typing
from typing import Optional, Callable
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
@property
def ordering_key(self) -> str:
    """The ordering key used to publish the message."""
    return self._ordering_key