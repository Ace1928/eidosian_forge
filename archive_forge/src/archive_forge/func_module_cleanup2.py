import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def module_cleanup2(*args, **kwargs):
    module_cleanups.append((4, args, kwargs))