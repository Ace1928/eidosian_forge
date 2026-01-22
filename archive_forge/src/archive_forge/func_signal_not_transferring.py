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
def signal_not_transferring(self):
    self.disable_callback()
    if hasattr(self._fileobj, 'signal_not_transferring'):
        self._fileobj.signal_not_transferring()