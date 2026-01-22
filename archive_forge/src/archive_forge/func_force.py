import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
def force(self):
    """Force send any remaining reports."""
    for address, msg in self.r_requests.iteritems():
        try:
            self.send(msg, address)
        except:
            continue
    for address, msg in self.w_requests.iteritems():
        try:
            self.send(msg, address)
        except:
            continue