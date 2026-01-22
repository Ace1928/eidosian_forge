import contextlib
import functools
import logging
import os
import sys
import time
import traceback
from kazoo import client
from taskflow.conductors import backends as conductor_backends
from taskflow import engines
from taskflow.jobs import backends as job_backends
from taskflow import logging as taskflow_logging
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends as persistence_backends
from taskflow.persistence import models
from taskflow import task
from oslo_utils import timeutils
from oslo_utils import uuidutils
def check_for_zookeeper(timeout=1):
    sys.stderr.write('Testing for the existence of a zookeeper server...\n')
    sys.stderr.write('Please wait....\n')
    with contextlib.closing(client.KazooClient()) as test_client:
        try:
            test_client.start(timeout=timeout)
        except test_client.handler.timeout_exception:
            sys.stderr.write('Zookeeper is needed for running this example!\n')
            traceback.print_exc()
            return False
        else:
            test_client.stop()
            return True