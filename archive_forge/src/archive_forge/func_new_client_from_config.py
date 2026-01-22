import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
def new_client_from_config(config_file=None, context=None, persist_config=True):
    """
    Loads configuration the same as load_kube_config but returns an ApiClient
    to be used with any API object. This will allow the caller to concurrently
    talk with multiple clusters.
    """
    client_config = type.__call__(Configuration)
    load_kube_config(config_file=config_file, context=context, client_configuration=client_config, persist_config=persist_config)
    return ApiClient(configuration=client_config)