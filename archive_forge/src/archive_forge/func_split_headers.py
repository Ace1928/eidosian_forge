import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
def split_headers(options, prefix=''):
    """
    Splits 'Key: Value' strings and returns them as a dictionary.

    :param options: Must be one of:
        * an iterable of 'Key: Value' strings
        * an iterable of ('Key', 'Value') pairs
        * a dict of {'Key': 'Value'} pairs
    :param prefix: String to prepend to all of the keys in the dictionary.
        reporting.
    """
    headers = {}
    try:
        headers = split_request_headers(options, prefix)
    except ValueError as e:
        raise SwiftError(e)
    return headers