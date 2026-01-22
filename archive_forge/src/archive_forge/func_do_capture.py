import asyncore
import errno
import socket
import threading
from taskflow.engines.action_engine import process_executor as pu
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
def do_capture(identity, message_capture_func):
    capture_buf.append(message_capture_func())