import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def get_worker_count():
    """Utility to get the default worker count.

    :returns: The number of CPUs if that can be determined, else a default
              worker count of 1 is returned.
    """
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1