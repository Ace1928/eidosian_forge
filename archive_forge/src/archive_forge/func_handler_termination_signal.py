from __future__ import annotations
import errno
import logging
import os
import os.path
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from io import StringIO
from textwrap import dedent
from typing import TYPE_CHECKING
from watchdog.observers.api import BaseObserverSubclassCallable
from watchdog.utils import WatchdogShutdown, load_class
from watchdog.version import VERSION_STRING
def handler_termination_signal(_signum, _frame):
    for signum in termination_signals:
        signal.signal(signum, signal.SIG_IGN)
    raise WatchdogShutdown