import os
import sys
import threading
from itertools import count
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from time import sleep
from kombu.asynchronous.timer import Entry
from kombu.asynchronous.timer import Timer as Schedule
from kombu.asynchronous.timer import logger, to_timestamp
``bool(timer)``.