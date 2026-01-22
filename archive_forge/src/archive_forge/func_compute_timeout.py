import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
def compute_timeout(offset):
    return math.ceil(time.monotonic() + offset)