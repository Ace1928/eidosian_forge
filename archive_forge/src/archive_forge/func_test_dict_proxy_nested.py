import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
def test_dict_proxy_nested(self):
    pets = self.dict(ferrets=2, hamsters=4)
    supplies = self.dict(water=10, feed=3)
    d = self.dict(pets=pets, supplies=supplies)
    self.assertEqual(supplies['water'], 10)
    self.assertEqual(d['supplies']['water'], 10)
    d['supplies']['blankets'] = 5
    self.assertEqual(supplies['blankets'], 5)
    self.assertEqual(d['supplies']['blankets'], 5)
    d['supplies']['water'] = 7
    self.assertEqual(supplies['water'], 7)
    self.assertEqual(d['supplies']['water'], 7)
    del pets
    del supplies
    self.assertEqual(d['pets']['ferrets'], 2)
    d['supplies']['blankets'] = 11
    self.assertEqual(d['supplies']['blankets'], 11)
    pets = d['pets']
    supplies = d['supplies']
    supplies['water'] = 7
    self.assertEqual(supplies['water'], 7)
    self.assertEqual(d['supplies']['water'], 7)
    d.clear()
    self.assertEqual(len(d), 0)
    self.assertEqual(supplies['water'], 7)
    self.assertEqual(pets['hamsters'], 4)
    l = self.list([pets, supplies])
    l[0]['marmots'] = 1
    self.assertEqual(pets['marmots'], 1)
    self.assertEqual(l[0]['marmots'], 1)
    del pets
    del supplies
    self.assertEqual(l[0]['marmots'], 1)
    outer = self.list([[88, 99], l])
    self.assertIsInstance(outer[0], list)
    self.assertEqual(outer[-1][-1]['feed'], 3)