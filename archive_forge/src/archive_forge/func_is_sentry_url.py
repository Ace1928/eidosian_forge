import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def is_sentry_url(hub, url):
    """
    Determines whether the given URL matches the Sentry DSN.
    """
    return hub.client is not None and hub.client.transport is not None and (hub.client.transport.parsed_dsn is not None) and (hub.client.transport.parsed_dsn.netloc in url)