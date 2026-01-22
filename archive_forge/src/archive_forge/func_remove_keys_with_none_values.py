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
def remove_keys_with_none_values(d: Union[Dict[str, Any], Any]) -> Union[Dict[str, Any], Any]:
    if not isinstance(d, dict):
        return d
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_v = remove_keys_with_none_values(v)
            if new_v is not None and (not (isinstance(new_v, dict) and len(new_v) == 0)):
                new_dict[k] = new_v
        return new_dict if new_dict else None