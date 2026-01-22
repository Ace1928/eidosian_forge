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
@normalize_exceptions
def register_agent(self, host: str, sweep_id: Optional[str]=None, project_name: Optional[str]=None, entity: Optional[str]=None) -> dict:
    """Register a new agent.

        Arguments:
            host (str): hostname
            sweep_id (str): sweep id
            project_name: (str): model that contains sweep
            entity: (str): entity that contains sweep
        """
    mutation = gql('\n        mutation CreateAgent(\n            $host: String!\n            $projectName: String,\n            $entityName: String,\n            $sweep: String!\n        ) {\n            createAgent(input: {\n                host: $host,\n                projectName: $projectName,\n                entityName: $entityName,\n                sweep: $sweep,\n            }) {\n                agent {\n                    id\n                }\n            }\n        }\n        ')
    if entity is None:
        entity = self.settings('entity')
    if project_name is None:
        project_name = self.settings('project')
    response = self.gql(mutation, variable_values={'host': host, 'entityName': entity, 'projectName': project_name, 'sweep': sweep_id}, check_retry_fn=util.no_retry_4xx)
    result: dict = response['createAgent']['agent']
    return result