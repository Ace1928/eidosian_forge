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
def create_default_resource_config(self, entity: str, resource: str, config: str, template_variables: Optional[Dict[str, Union[float, int, str]]]) -> Optional[Dict[str, Any]]:
    if not self.create_default_resource_config_introspection():
        raise Exception()
    supports_template_vars, _ = self.push_to_run_queue_introspection()
    mutation_params = '\n            $entityName: String!,\n            $resource: String!,\n            $config: JSONString!\n        '
    mutation_inputs = '\n            entityName: $entityName,\n            resource: $resource,\n            config: $config\n        '
    if supports_template_vars:
        mutation_params += ', $templateVariables: JSONString'
        mutation_inputs += ', templateVariables: $templateVariables'
    elif template_variables is not None:
        raise UnsupportedError('server does not support template variables, please update server instance to >=0.46')
    variable_values = {'entityName': entity, 'resource': resource, 'config': config}
    if supports_template_vars:
        if template_variables is not None:
            variable_values['templateVariables'] = json.dumps(template_variables)
        else:
            variable_values['templateVariables'] = '{}'
    query = gql(f'\n        mutation createDefaultResourceConfig(\n            {mutation_params}\n        ) {{\n            createDefaultResourceConfig(\n            input: {{\n                {mutation_inputs}\n            }}\n            ) {{\n            defaultResourceConfigID\n            success\n            }}\n        }}\n        ')
    result: Optional[Dict[str, Any]] = self.gql(query, variable_values)['createDefaultResourceConfig']
    return result