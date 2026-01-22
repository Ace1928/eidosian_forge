import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def redirect_log_thread_fn():
    for line in process.stdout:
        tail_output_deque.append(line)
        sys.stdout.write(line)