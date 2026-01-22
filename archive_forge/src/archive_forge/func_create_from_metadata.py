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
@classmethod
def create_from_metadata(cls, metadata, refresh_using, method, advisory_timeout=None, mandatory_timeout=None):
    kwargs = {}
    if advisory_timeout is not None:
        kwargs['advisory_timeout'] = advisory_timeout
    if mandatory_timeout is not None:
        kwargs['mandatory_timeout'] = mandatory_timeout
    instance = cls(access_key=metadata['access_key'], secret_key=metadata['secret_key'], token=metadata['token'], expiry_time=cls._expiry_datetime(metadata['expiry_time']), method=method, refresh_using=refresh_using, **kwargs)
    return instance