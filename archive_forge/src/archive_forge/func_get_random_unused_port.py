import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def get_random_unused_port(host, min_port=1024, max_port=65535, max_retries=100, exclude_list=None):
    """
    Get random unused port.
    """
    rng = random.SystemRandom()
    exclude_list = exclude_list or []
    for _ in range(max_retries):
        port = rng.randint(min_port, max_port)
        if port in exclude_list:
            continue
        if not is_port_in_use(host, port):
            return port
    raise RuntimeError(f'Get available port between range {min_port} and {max_port} failed.')