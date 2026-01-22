import time
import socket
import argparse
import sys
import itertools
import contextlib
import platform
from collections import abc
import urllib.parse
from tempora import timing
@classmethod
def from_addr(cls, addr):
    listen_host, port = addr[:2]
    plain_host = client_host(listen_host)
    host = f'[{plain_host}]' if ':' in plain_host else plain_host
    return cls(':'.join([host, str(port)]))