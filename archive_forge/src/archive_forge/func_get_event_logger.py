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
def get_event_logger(source: Event.SourceType, sink_dir: str):
    """Get the event logger of the current process.

    There's only 1 event logger per (process, source).

    TODO(sang): Support more impl than file-based logging.
                Currently, the interface also ties to the
                file-based logging impl.

    Args:
        source: The source of the event.
        sink_dir: The directory to sink event logs.
    """
    with _event_logger_lock:
        global _event_logger
        source_name = Event.SourceType.Name(source)
        if source_name not in _event_logger:
            logger = _build_event_file_logger(source_name, sink_dir)
            _event_logger[source_name] = EventLoggerAdapter(source, logger)
        return _event_logger[source_name]