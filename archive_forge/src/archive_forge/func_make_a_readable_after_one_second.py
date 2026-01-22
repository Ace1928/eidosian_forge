import signal
import socket
import threading
import time
import pytest
from urllib3.util.wait import (
from .socketpair_helper import socketpair
def make_a_readable_after_one_second():
    time.sleep(1)
    b.send(b'x')