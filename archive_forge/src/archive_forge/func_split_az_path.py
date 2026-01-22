import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def split_az_path(path: str) -> Tuple[str, str, str]:
    parts = path[len('az://'):].split('/')
    if len(parts) < 2:
        raise Error(f"Invalid path: '{path}'")
    account = parts[0]
    container = parts[1]
    if container == '':
        raise Error(f"Invalid path: '{path}'")
    obj = '/'.join(parts[2:])
    return (account, container, obj)