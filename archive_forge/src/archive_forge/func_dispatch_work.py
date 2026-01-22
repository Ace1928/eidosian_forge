import collections
import contextlib
import logging
import os
import random
import sys
import threading
import time
from zake import fake_client
from taskflow import exceptions as excp
from taskflow.jobs import backends
from taskflow.utils import threading_utils
def dispatch_work(job):
    time.sleep(1.0)