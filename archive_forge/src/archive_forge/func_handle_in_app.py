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
def handle_in_app(event, in_app_exclude=None, in_app_include=None, project_root=None):
    for stacktrace in iter_event_stacktraces(event):
        set_in_app_in_frames(stacktrace.get('frames'), in_app_exclude=in_app_exclude, in_app_include=in_app_include, project_root=project_root)
    return event