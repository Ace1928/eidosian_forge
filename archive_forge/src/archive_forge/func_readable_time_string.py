import contextlib
from datetime import datetime
import sys
import time
def readable_time_string():
    """Get a human-readable time string for the present."""
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')