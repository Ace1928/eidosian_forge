import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def process_message(self, peer, mailfrom, rcpttos, data):
    lines = data.split('\n')
    i = 0
    for line in lines:
        if not line:
            break
        i += 1
    lines.insert(i, 'X-Peer: %s' % peer[0])
    data = NEWLINE.join(lines)
    refused = self._deliver(mailfrom, rcpttos, data)
    print('we got some refusals:', refused, file=DEBUGSTREAM)