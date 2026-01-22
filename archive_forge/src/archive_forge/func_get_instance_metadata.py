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
def get_instance_metadata(version='latest', url='http://169.254.169.254', data='meta-data/', timeout=None, num_retries=5):
    """
    Returns the instance metadata as a nested Python dictionary.
    Simple values (e.g. local_hostname, hostname, etc.) will be
    stored as string values.  Values such as ancestor-ami-ids will
    be stored in the dict as a list of string values.  More complex
    fields such as public-keys and will be stored as nested dicts.

    If the timeout is specified, the connection to the specified url
    will time out after the specified number of seconds.

    """
    try:
        metadata_url = _build_instance_metadata_url(url, version, data)
        return _get_instance_metadata(metadata_url, num_retries=num_retries, timeout=timeout)
    except urllib.error.URLError:
        boto.log.exception('Exception caught when trying to retrieve instance metadata for: %s', data)
        return None