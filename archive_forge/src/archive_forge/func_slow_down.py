import contextlib
import hashlib
import logging
import os
import random
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils  # noqa
@contextlib.contextmanager
def slow_down(how_long=0.5):
    try:
        yield how_long
    finally:
        print('** Ctrl-c me please!!! **')
        time.sleep(how_long)