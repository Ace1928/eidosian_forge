import random
import threading
import time
from .messages import Message
from .parser import Parser
def set_sleep_time(seconds=_DEFAULT_SLEEP_TIME):
    """Set the number of seconds sleep() will sleep."""
    global _sleep_time
    _sleep_time = seconds