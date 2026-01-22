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
def upload_file_azure(self, url: str, file: Any, extra_headers: Dict[str, str]) -> None:
    """Upload a file to azure."""
    from azure.core.exceptions import AzureError
    client = self._azure_blob_module.BlobClient.from_blob_url(url, retry_policy=self._azure_blob_module.LinearRetry(retry_total=0))
    try:
        if extra_headers.get('Content-MD5') is not None:
            md5: Optional[bytes] = base64.b64decode(extra_headers['Content-MD5'])
        else:
            md5 = None
        content_settings = self._azure_blob_module.ContentSettings(content_md5=md5, content_type=extra_headers.get('Content-Type'))
        client.upload_blob(file, max_concurrency=4, length=len(file), overwrite=True, content_settings=content_settings)
    except AzureError as e:
        if hasattr(e, 'response'):
            response = requests.models.Response()
            response.status_code = e.response.status_code
            response.headers = e.response.headers
            raise requests.exceptions.RequestException(e.message, response=response)
        else:
            raise requests.exceptions.ConnectionError(e.message)