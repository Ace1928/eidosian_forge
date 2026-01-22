import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
def launch_queue_logger_thread(output_producer, should_end):

    def queue_logger_thread(out_prod, should_end):
        while not should_end():
            try:
                line, running = out_prod.get_output()
                if not running:
                    break
                if line:
                    level = line[0]
                    record = line[1]
                    name = line[2]
                    lgr = logging.getLogger(name)
                    lgr.log(level, record)
            except Exception as e:
                print(e)
                break
    thread = threading.Thread(target=queue_logger_thread, args=(output_producer, should_end))
    thread.setDaemon(True)
    thread.start()