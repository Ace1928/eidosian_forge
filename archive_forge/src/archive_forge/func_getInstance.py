from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def getInstance():
    """ Static access method. """
    if nvidia_smi.__instance == None:
        nvidia_smi()
    return nvidia_smi.__instance