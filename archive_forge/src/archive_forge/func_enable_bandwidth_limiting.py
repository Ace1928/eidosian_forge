import time
import threading
def enable_bandwidth_limiting(self):
    """Enable bandwidth limiting on reads to the stream"""
    self._bandwidth_limiting_enabled = True