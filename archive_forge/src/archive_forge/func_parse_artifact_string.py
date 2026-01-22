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
def parse_artifact_string(v: str) -> Tuple[str, Optional[str], bool]:
    if not v.startswith('wandb-artifact://'):
        raise ValueError(f'Invalid artifact string: {v}')
    parsed_v = v[len('wandb-artifact://'):]
    base_uri = None
    url_info = urllib.parse.urlparse(parsed_v)
    if url_info.scheme != '':
        base_uri = f'{url_info.scheme}://{url_info.netloc}'
        parts = url_info.path.split('/')[1:]
    else:
        parts = parsed_v.split('/')
    if parts[0] == '_id':
        return (parts[1], base_uri, True)
    if len(parts) < 3:
        raise ValueError(f'Invalid artifact string: {v}')
    entity, project, name_and_alias_or_version = parts[:3]
    return (f'{entity}/{project}/{name_and_alias_or_version}', base_uri, False)