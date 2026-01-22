import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def tcp_client(self, client_prog, family=socket.AF_INET, timeout=10):
    sock = socket.socket(family, socket.SOCK_STREAM)
    if timeout is None:
        raise RuntimeError('timeout is required')
    if timeout <= 0:
        raise RuntimeError('only blocking sockets are supported')
    sock.settimeout(timeout)
    return TestThreadedClient(self, sock, client_prog, timeout)