import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
def gcf():

    def inner():
        ctx = self.Context()
        ctx.socket(zmq.PUSH)
    inner()
    gc.collect()