import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def json_friendly_val(val: Any) -> Any:
    """Make any value (including dict, slice, sequence, dataclass) JSON friendly."""
    converted: Union[dict, list]
    if isinstance(val, dict):
        converted = {}
        for key, value in val.items():
            converted[key] = json_friendly_val(value)
        return converted
    if isinstance(val, slice):
        converted = dict(slice_start=val.start, slice_step=val.step, slice_stop=val.stop)
        return converted
    val, _ = json_friendly(val)
    if isinstance(val, Sequence) and (not isinstance(val, str)):
        converted = []
        for value in val:
            converted.append(json_friendly_val(value))
        return converted
    if is_dataclass(val) and (not isinstance(val, type)):
        converted = asdict(val)
        return converted
    else:
        if val.__class__.__module__ not in ('builtins', '__builtin__'):
            val = str(val)
        return val