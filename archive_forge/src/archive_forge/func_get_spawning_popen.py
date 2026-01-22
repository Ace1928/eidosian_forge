import os
import sys
import threading
from . import process
from . import reduction
def get_spawning_popen():
    return getattr(_tls, 'spawning_popen', None)