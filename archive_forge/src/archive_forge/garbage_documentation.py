import atexit
import struct
import warnings
from collections import namedtuple
from os import getpid
from threading import Event, Lock, Thread
import zmq
store an object and (optionally) event for zero-copy