import os
import sys
import signal
import pickle
from io import BytesIO
from multiprocessing import util, process
from multiprocessing.connection import wait
from multiprocessing.context import set_spawning_popen
from . import reduction, resource_tracker, spawn
@staticmethod
def thread_is_spawning():
    return True