import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
def graceful(self):
    """Advise all services to reload."""
    self.log('Bus graceful')
    self.publish('graceful')