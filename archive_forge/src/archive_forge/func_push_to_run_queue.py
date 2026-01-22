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
def push_to_run_queue(self, queue_name: str, launch_spec: Dict[str, str], template_variables: Optional[dict], project_queue: str, priority: Optional[int]=None) -> Optional[Dict[str, Any]]:
    self.push_to_run_queue_introspection()
    entity = launch_spec.get('queue_entity') or launch_spec['entity']
    run_spec = json.dumps(launch_spec)
    push_result = self.push_to_run_queue_by_name(entity, project_queue, queue_name, run_spec, template_variables, priority)
    if push_result:
        return push_result
    if priority is not None:
        return None
    ' Legacy Method '
    queues_found = self.get_project_run_queues(entity, project_queue)
    matching_queues = [q for q in queues_found if q['name'] == queue_name and (q['access'] in ['PROJECT', 'USER'] or q['createdBy'] == self.default_entity)]
    if not matching_queues:
        if queue_name == 'default':
            wandb.termlog(f'No default queue existing for entity: {entity} in project: {project_queue}, creating one.')
            res = self.create_run_queue(launch_spec['entity'], project_queue, queue_name, access='PROJECT')
            if res is None or res.get('queueID') is None:
                wandb.termerror(f'Unable to create default queue for entity: {entity} on project: {project_queue}. Run could not be added to a queue')
                return None
            queue_id = res['queueID']
        else:
            if project_queue == 'model-registry':
                _msg = f'Unable to push to run queue {queue_name}. Queue not found.'
            else:
                _msg = f'Unable to push to run queue {project_queue}/{queue_name}. Queue not found.'
            wandb.termwarn(_msg)
            return None
    elif len(matching_queues) > 1:
        wandb.termerror(f'Unable to push to run queue {queue_name}. More than one queue found with this name.')
        return None
    else:
        queue_id = matching_queues[0]['id']
    spec_json = json.dumps(launch_spec)
    variables = {'queueID': queue_id, 'runSpec': spec_json}
    mutation_params = '\n            $queueID: ID!,\n            $runSpec: JSONString!\n        '
    mutation_input = '\n            queueID: $queueID,\n            runSpec: $runSpec\n        '
    if self.server_supports_template_variables:
        if template_variables is not None:
            mutation_params += ', $templateVariableValues: JSONString'
            mutation_input += ', templateVariableValues: $templateVariableValues'
            variables.update({'templateVariableValues': json.dumps(template_variables)})
    elif template_variables is not None:
        raise UnsupportedError('server does not support template variables, please update server instance to >=0.46')
    mutation = gql(f'\n        mutation pushToRunQueue(\n            {mutation_params}\n            ) {{\n            pushToRunQueue(\n                input: {{{mutation_input}}}\n            ) {{\n                runQueueItemId\n            }}\n        }}\n        ')
    response = self.gql(mutation, variable_values=variables)
    if not response.get('pushToRunQueue'):
        raise CommError(f'Error pushing run queue item to queue {queue_name}.')
    result: Optional[Dict[str, Any]] = response['pushToRunQueue']
    return result