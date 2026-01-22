import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
def parse_ts(ts):
    with setlocale('C'):
        ts = ts.strip()
        try:
            dt = datetime.datetime.strptime(ts, ISO8601)
            return dt
        except ValueError:
            try:
                dt = datetime.datetime.strptime(ts, ISO8601_MS)
                return dt
            except ValueError:
                dt = datetime.datetime.strptime(ts, RFC1123)
                return dt