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
def check_client_connected(client_pid, keep_alive_proxy):
    logger.debug('Client keep-alive connection monitor started for {}.'.format(client_pid))
    while True:
        time.sleep(InstanceManager.KEEP_ALIVE_PYRO_FREQUENCY)
        try:
            keep_alive_proxy.call()
        except:
            bad_insts = [inst for inst in cls._instance_pool if inst.owner == client_pid]
            for inst in bad_insts:
                inst.close()