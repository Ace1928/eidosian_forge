import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
start a timer to fire only once

        like signal.alarm, but with better resolution than integer seconds.
        