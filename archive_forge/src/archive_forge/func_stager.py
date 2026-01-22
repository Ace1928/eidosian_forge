import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
def stager():
    stager.stage += 1
    if stager.stage < 3:
        launcher._handle_hup(1, mock.sentinel.frame)
    elif stager.stage == 3:
        launcher._handle_term(15, mock.sentinel.frame)
    else:
        self.fail('TERM did not kill launcher')