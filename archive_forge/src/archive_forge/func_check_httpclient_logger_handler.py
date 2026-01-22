import ast
import asyncio
import base64
import datetime
import functools
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
from copy import deepcopy
from typing import (
import click
import requests
import yaml
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis.normalize import normalize_exceptions, parse_backend_error_messages
from wandb.errors import CommError, UnsupportedError, UsageError
from wandb.integration.sagemaker import parse_sm_secrets
from wandb.old.settings import Settings
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.gql_request import GraphQLSession
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64
from ..lib import retry
from ..lib.filenames import DIFF_FNAME, METADATA_FNAME
from ..lib.gitlib import GitRepo
from . import context
from .progress import AsyncProgress, Progress
def check_httpclient_logger_handler() -> None:
    if not os.environ.get('WANDB_DEBUG'):
        return
    if httpclient_logger.handlers:
        return
    level = logging.DEBUG

    def httpclient_log(*args: Any) -> None:
        httpclient_logger.log(level, ' '.join(args))
    http.client.print = httpclient_log
    http.client.HTTPConnection.debuglevel = 1
    root_logger = logging.getLogger('wandb')
    if root_logger.handlers:
        httpclient_logger.addHandler(root_logger.handlers[0])