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
def image_id_from_k8s() -> Optional[str]:
    """Ping the k8s metadata service for the image id.

    Specify the KUBERNETES_NAMESPACE environment variable if your pods are not in the
    default namespace:

    - name: KUBERNETES_NAMESPACE valueFrom:
        fieldRef:
          fieldPath: metadata.namespace
    """
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    if not os.path.exists(token_path):
        return None
    try:
        with open(token_path) as token_file:
            token = token_file.read()
    except FileNotFoundError:
        logger.warning(f'Token file not found at {token_path}.')
        return None
    except PermissionError as e:
        current_uid = os.getuid()
        warning = f'Unable to read the token file at {token_path} due to permission error ({e}).The current user id is {current_uid}. Consider changing the securityContext to run the container as the current user.'
        logger.warning(warning)
        wandb.termwarn(warning)
        return None
    if not token:
        return None
    k8s_server = 'https://{}:{}/api/v1/namespaces/{}/pods/{}'.format(os.getenv('KUBERNETES_SERVICE_HOST'), os.getenv('KUBERNETES_PORT_443_TCP_PORT'), os.getenv('KUBERNETES_NAMESPACE', 'default'), os.getenv('HOSTNAME'))
    try:
        res = requests.get(k8s_server, verify='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt', timeout=3, headers={'Authorization': f'Bearer {token}'})
        res.raise_for_status()
    except requests.RequestException:
        return None
    try:
        return str(res.json()['status']['containerStatuses'][0]['imageID']).strip('docker-pullable://')
    except (ValueError, KeyError, IndexError):
        logger.exception('Error checking kubernetes for image id')
        return None