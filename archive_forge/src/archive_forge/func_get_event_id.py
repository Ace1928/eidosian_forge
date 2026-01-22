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
def get_event_id():
    return ''.join([random.choice(string.hexdigits) for _ in range(36)])