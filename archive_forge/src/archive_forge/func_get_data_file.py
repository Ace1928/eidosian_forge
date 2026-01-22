import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def get_data_file(args):
    if args.file:
        return open(args.file, 'rb')
    else:
        try:
            os.fstat(0)
        except OSError:
            return None
        if hasattr(sys.stdin, 'isatty') and (not sys.stdin.isatty()):
            image = sys.stdin
            if hasattr(sys.stdin, 'buffer'):
                image = sys.stdin.buffer
            if msvcrt:
                msvcrt.setmode(image.fileno(), os.O_BINARY)
            return image
        else:
            return None