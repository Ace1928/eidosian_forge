import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase
from pytest import mark
import zmq
from zmq.utils import jsonapi
def ping_pong_json(self, s1, s2, o):
    if jsonapi.jsonmod is None:
        raise SkipTest('No json library')
    s1.send_json(o)
    o2 = s2.recv_json()
    s2.send_json(o2)
    o3 = s1.recv_json()
    return o3