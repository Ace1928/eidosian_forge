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
def launch_instance_manager():
    """Defines the entry point for the remote procedure call server.
    """
    parser = argparse.ArgumentParser('python3 launch_instance_manager.py')
    parser.add_argument('--seeds', type=str, default=None, help='The default seed for the environment.')
    parser.add_argument('--seeding_type', type=str, default=SeedType.CONSTANT, help='The seeding type for the environment. Defaults to 1 (CONSTANT)if a seed specified, otherwise 0 (NONE): \n{}'.format(SeedType.__doc__))
    parser.add_argument('--max_instances', type=int, default=None, help='The maximum number of instances the instance manager is able to spawn,before an exception is thrown. Defaults to Unlimited.')
    opts = parser.parse_args()
    if opts.max_instances is not None:
        assert opts.max_instances > 0, 'Maximum instances must be more than zero!'
        InstanceManager.MAXINSTANCES = opts.max_instances
    try:
        print('Removing the performance directory!')
        try:
            shutil.rmtree(InstanceManager.STATUS_DIR)
        except:
            pass
        finally:
            if not os.path.exists(InstanceManager.STATUS_DIR):
                os.makedirs(InstanceManager.STATUS_DIR)
        print('autoproxy?', Pyro4.config.AUTOPROXY)
        InstanceManager.REMOTE = True
        Pyro4.config.COMMTIMEOUT = InstanceManager.KEEP_ALIVE_PYRO_FREQUENCY
        if opts.seeds is not None:
            InstanceManager._init_seeding(seeds=opts.seeds, seed_type=opts.seeding_type)
        else:
            InstanceManager._init_seeding(seed_type=SeedType.NONE)
        Pyro4.Daemon.serveSimple({InstanceManager: INSTANCE_MANAGER_PYRO}, ns=True)
    except Pyro4.errors.NamingError as e:
        print(e)
        print('Start the Pyro name server with pyro4-ns and re-run this script.')