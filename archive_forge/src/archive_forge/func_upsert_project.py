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
def upsert_project(self, project: str, id: Optional[str]=None, description: Optional[str]=None, entity: Optional[str]=None) -> Dict[str, Any]:
    """Create a new project.

        Arguments:
            project (str): The project to create
            description (str, optional): A description of this project
            entity (str, optional): The entity to scope this project to.
        """
    mutation = gql('\n        mutation UpsertModel($name: String!, $id: String, $entity: String!, $description: String, $repo: String)  {\n            upsertModel(input: { id: $id, name: $name, entityName: $entity, description: $description, repo: $repo }) {\n                model {\n                    name\n                    description\n                }\n            }\n        }\n        ')
    response = self.gql(mutation, variable_values={'name': self.format_project(project), 'entity': entity or self.settings('entity'), 'description': description, 'id': id})
    result: Dict[str, Any] = response['upsertModel']['model']
    return result