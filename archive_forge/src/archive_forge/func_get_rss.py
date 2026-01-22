import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
def get_rss(memory_info):
    """Get the estimated non-shared memory usage from psutil memory_info."""
    mem = memory_info.rss
    if hasattr(memory_info, 'shared'):
        mem -= memory_info.shared
    return mem