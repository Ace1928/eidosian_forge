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
def log_to_file(logdir):
    if not os.path.exists(os.path.join(logdir, 'logs')):
        os.makedirs(os.path.join(logdir, 'logs'))
    file_path = os.path.join(logdir, 'logs', 'mc_{}.log'.format(self._target_port - 9000))
    logger.info('Logging output of Minecraft to {}'.format(file_path))
    mine_log = open(file_path, 'wb+')
    mine_log.truncate(0)
    mine_log_encoding = locale.getpreferredencoding(False)
    try:
        while self.running:
            line = self.minecraft_process.stdout.readline()
            if not line:
                break
            try:
                linestr = line.decode(mine_log_encoding)
            except UnicodeDecodeError:
                mine_log_encoding = locale.getpreferredencoding(False)
                logger.error('UnicodeDecodeError, switching to default encoding')
                linestr = line.decode(mine_log_encoding)
            linestr = '\n'.join(linestr.split('\n')[:-1])
            self._log_heuristic(linestr)
            mine_log.write(line)
            mine_log.flush()
    finally:
        mine_log.close()