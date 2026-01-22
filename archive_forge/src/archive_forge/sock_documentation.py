import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util

    Create a new socket for the configured addresses or file descriptors.

    If a configured address is a tuple then a TCP socket is created.
    If it is a string, a Unix socket is created. Otherwise, a TypeError is
    raised.
    