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
def parse_host(hostname):
    """
    Given a hostname that may have a port name, ensure that the port is trimmed
    returning only the host, including hostnames that are IPV6 and may include
    brackets.
    """
    hostname = hostname.strip()
    if host_is_ipv6(hostname):
        return hostname.split(']:', 1)[0].strip('[]')
    else:
        return hostname.split(':', 1)[0]