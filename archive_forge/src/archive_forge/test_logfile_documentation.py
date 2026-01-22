from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase

        Test that L{DailyLogFile.toDate} uses its arguments to create a new
        date.
        