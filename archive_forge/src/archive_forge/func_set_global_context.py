import logging
import pathlib
import json
import random
import string
import socket
import os
import threading
from typing import Dict, Optional
from datetime import datetime
from google.protobuf.json_format import Parse
from ray.core.generated.event_pb2 import Event
from ray._private.protobuf_compat import message_to_dict
def set_global_context(self, global_context: Dict[str, str]=None):
    """Set the global metadata.

        This method overwrites the global metadata if it is called more than once.
        """
    with self.lock:
        self.global_context = {} if not global_context else global_context