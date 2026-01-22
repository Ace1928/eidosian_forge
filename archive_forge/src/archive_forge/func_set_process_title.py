from __future__ import annotations
import numbers
import socket
import sys
from datetime import datetime
from signal import Signals
from types import FrameType
from typing import Any
from celery import VERSION_BANNER, Celery, beat, platforms
from celery.utils.imports import qualname
from celery.utils.log import LOG_LEVELS, get_logger
from celery.utils.time import humanize_seconds
def set_process_title(self) -> None:
    arg_start = 'manage' in sys.argv[0] and 2 or 1
    platforms.set_process_title('celery beat', info=' '.join(sys.argv[arg_start:]))