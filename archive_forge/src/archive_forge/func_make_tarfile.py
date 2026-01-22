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
def make_tarfile(output_filename: str, source_dir: str, archive_name: str, custom_filter: Optional[Callable]=None) -> None:

    def _filter_timestamps(tar_info: 'tarfile.TarInfo') -> Optional['tarfile.TarInfo']:
        tar_info.mtime = 0
        return tar_info if custom_filter is None else custom_filter(tar_info)
    descriptor, unzipped_filename = tempfile.mkstemp()
    try:
        with tarfile.open(unzipped_filename, 'w') as tar:
            tar.add(source_dir, arcname=archive_name, filter=_filter_timestamps)
        with gzip.GzipFile(filename='', fileobj=open(output_filename, 'wb'), mode='wb', mtime=0) as gzipped_tar, open(unzipped_filename, 'rb') as tar_file:
            gzipped_tar.write(tar_file.read())
    finally:
        os.close(descriptor)
        os.remove(unzipped_filename)