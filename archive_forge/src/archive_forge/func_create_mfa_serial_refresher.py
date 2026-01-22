import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
def create_mfa_serial_refresher(actual_refresh):

    class _Refresher:

        def __init__(self, refresh):
            self._refresh = refresh
            self._has_been_called = False

        def __call__(self):
            if self._has_been_called:
                raise RefreshWithMFAUnsupportedError()
            self._has_been_called = True
            return self._refresh()
    return _Refresher(actual_refresh)