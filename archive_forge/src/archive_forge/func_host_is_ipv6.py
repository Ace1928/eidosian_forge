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
def host_is_ipv6(hostname):
    """
    Detect (naively) if the hostname is an IPV6 host.
    Return a boolean.
    """
    if not hostname or not isinstance(hostname, str):
        return False
    if hostname.startswith('['):
        return True
    if len(hostname.split(':')) > 2:
        return True
    return False