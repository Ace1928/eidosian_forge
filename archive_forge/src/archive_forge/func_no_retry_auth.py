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
def no_retry_auth(e: Any) -> bool:
    if hasattr(e, 'exception'):
        e = e.exception
    if not isinstance(e, requests.HTTPError):
        return True
    if e.response is None:
        return True
    if e.response.status_code in (400, 409):
        return False
    if e.response.status_code not in (401, 403, 404):
        return True
    if e.response.status_code == 401:
        raise AuthenticationError(f"The API key you provided is either invalid or missing.  If the `{wandb.env.API_KEY}` environment variable is set, make sure it is correct. Otherwise, to resolve this issue, you may try running the 'wandb login --relogin' command. If you are using a local server, make sure that you're using the correct hostname. If you're not sure, you can try logging in again using the 'wandb login --relogin --host [hostname]' command.(Error {e.response.status_code}: {e.response.reason})")
    elif wandb.run:
        raise CommError(f'Permission denied to access {wandb.run.path}')
    else:
        raise CommError(f'It appears that you do not have permission to access the requested resource. Please reach out to the project owner to grant you access. If you have the correct permissions, verify that there are no issues with your networking setup.(Error {e.response.status_code}: {e.response.reason})')