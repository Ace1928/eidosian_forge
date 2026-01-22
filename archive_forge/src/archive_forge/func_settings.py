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
def settings(self, key: Optional[str]=None, section: Optional[str]=None) -> Any:
    """The settings overridden from the wandb/settings file.

        Arguments:
            key (str, optional): If provided only this setting is returned
            section (str, optional): If provided this section of the setting file is
            used, defaults to "default"

        Returns:
            A dict with the current settings

                {
                    "entity": "models",
                    "base_url": "https://api.wandb.ai",
                    "project": None
                }
        """
    result = self.default_settings.copy()
    result.update(self._settings.items(section=section))
    result.update({'entity': env.get_entity(self._settings.get(Settings.DEFAULT_SECTION, 'entity', fallback=result.get('entity')), env=self._environ), 'project': env.get_project(self._settings.get(Settings.DEFAULT_SECTION, 'project', fallback=result.get('project')), env=self._environ), 'base_url': env.get_base_url(self._settings.get(Settings.DEFAULT_SECTION, 'base_url', fallback=result.get('base_url')), env=self._environ), 'ignore_globs': env.get_ignore(self._settings.get(Settings.DEFAULT_SECTION, 'ignore_globs', fallback=result.get('ignore_globs')), env=self._environ)})
    return result if key is None else result[key]