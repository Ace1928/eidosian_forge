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
def parse_event(event_str: str) -> Optional[Event]:
    """Parse an event from a string.

    Args:
        event_str: The string to parse. Expect to be a JSON serialized
            Event protobuf.

    Returns:
        The parsed event if parsable, else None
    """
    try:
        return Parse(event_str, Event())
    except Exception:
        global_logger.exception(f'Failed to parse event: {event_str}')
        return None