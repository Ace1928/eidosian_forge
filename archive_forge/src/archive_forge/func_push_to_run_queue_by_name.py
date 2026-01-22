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
def push_to_run_queue_by_name(self, entity: str, project: str, queue_name: str, run_spec: str, template_variables: Optional[Dict[str, Union[int, float, str]]], priority: Optional[int]=None) -> Optional[Dict[str, Any]]:
    self.push_to_run_queue_introspection()
    'Queryless mutation, should be used before legacy fallback method.'
    mutation_params = '\n            $entityName: String!,\n            $projectName: String!,\n            $queueName: String!,\n            $runSpec: JSONString!\n        '
    mutation_input = '\n            entityName: $entityName,\n            projectName: $projectName,\n            queueName: $queueName,\n            runSpec: $runSpec\n        '
    variables: Dict[str, Any] = {'entityName': entity, 'projectName': project, 'queueName': queue_name, 'runSpec': run_spec}
    if self.server_push_to_run_queue_supports_priority:
        if priority is not None:
            variables['priority'] = priority
            mutation_params += ', $priority: Int'
            mutation_input += ', priority: $priority'
    elif priority is not None:
        raise UnsupportedError('server does not support priority, please update server instance to >=0.46')
    if self.server_supports_template_variables:
        if template_variables is not None:
            variables.update({'templateVariableValues': json.dumps(template_variables)})
            mutation_params += ', $templateVariableValues: JSONString'
            mutation_input += ', templateVariableValues: $templateVariableValues'
    elif template_variables is not None:
        raise UnsupportedError('server does not support template variables, please update server instance to >=0.46')
    mutation = gql(f'\n        mutation pushToRunQueueByName(\n          {mutation_params}\n        ) {{\n            pushToRunQueueByName(\n                input: {{\n                    {mutation_input}\n                }}\n            ) {{\n                runQueueItemId\n                runSpec\n            }}\n        }}\n        ')
    try:
        result: Optional[Dict[str, Any]] = self.gql(mutation, variables, check_retry_fn=util.no_retry_4xx).get('pushToRunQueueByName')
        if not result:
            return None
        if result.get('runSpec'):
            run_spec = json.loads(str(result['runSpec']))
            result['runSpec'] = run_spec
        return result
    except Exception as e:
        if 'Cannot query field "runSpec" on type "PushToRunQueueByNamePayload"' not in str(e):
            return None
    mutation_no_runspec = gql('\n        mutation pushToRunQueueByName(\n            $entityName: String!,\n            $projectName: String!,\n            $queueName: String!,\n            $runSpec: JSONString!,\n        ) {\n            pushToRunQueueByName(\n                input: {\n                    entityName: $entityName,\n                    projectName: $projectName,\n                    queueName: $queueName,\n                    runSpec: $runSpec\n                }\n            ) {\n                runQueueItemId\n            }\n        }\n        ')
    try:
        result = self.gql(mutation_no_runspec, variables, check_retry_fn=util.no_retry_4xx).get('pushToRunQueueByName')
    except Exception:
        result = None
    return result