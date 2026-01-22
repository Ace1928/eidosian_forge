import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
def get_filtered_dict(original_dict, whitelisted_keys):
    """Gets a dictionary filtered by whitelisted keys

    :param original_dict: The original dictionary of arguments to source keys
        and values.
    :param whitelisted_key: A list of keys to include in the filtered
        dictionary.

    :returns: A dictionary containing key/values from the original dictionary
        whose key was included in the whitelist
    """
    filtered_dict = {}
    for key, value in original_dict.items():
        if key in whitelisted_keys:
            filtered_dict[key] = value
    return filtered_dict