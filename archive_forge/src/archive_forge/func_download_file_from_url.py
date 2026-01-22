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
def download_file_from_url(dest_path: str, source_url: str, api_key: Optional[str]=None) -> None:
    auth = None
    if not _thread_local_api_settings.cookies:
        auth = ('api', api_key or '')
    response = requests.get(source_url, auth=auth, headers=_thread_local_api_settings.headers, cookies=_thread_local_api_settings.cookies, stream=True, timeout=5)
    response.raise_for_status()
    if os.sep in dest_path:
        filesystem.mkdir_exists_ok(os.path.dirname(dest_path))
    with fsync_open(dest_path, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)