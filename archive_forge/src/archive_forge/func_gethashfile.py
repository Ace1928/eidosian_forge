from __future__ import print_function
import os,stat,time
import errno
import sys
def gethashfile(key):
    return ('%02x' % abs(hash(key) % 256))[-2:]