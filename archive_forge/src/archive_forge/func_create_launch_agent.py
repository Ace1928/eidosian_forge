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
def create_launch_agent(self, entity: str, project: str, queues: List[str], agent_config: Dict[str, Any], version: str, gorilla_agent_support: bool) -> dict:
    project_queues = self.get_project_run_queues(entity, project)
    if not project_queues:
        default = self.create_run_queue(entity, project, 'default', access='PROJECT')
        if default is None or default.get('queueID') is None:
            raise CommError('Unable to create default queue for {}/{}. No queues for agent to poll'.format(entity, project))
        project_queues = [{'id': default['queueID'], 'name': 'default'}]
    polling_queue_ids = [q['id'] for q in project_queues if q['name'] in queues]
    if len(polling_queue_ids) != len(queues):
        raise CommError(f'Could not start launch agent: Not all of requested queues ({', '.join(queues)}) found. Available queues for this project: {','.join([q['name'] for q in project_queues])}')
    if not gorilla_agent_support:
        return {'success': True, 'launchAgentId': None}
    hostname = socket.gethostname()
    variable_values = {'entity': entity, 'project': project, 'queues': polling_queue_ids, 'hostname': hostname}
    mutation_params = '\n            $entity: String!,\n            $project: String!,\n            $queues: [ID!]!,\n            $hostname: String!\n        '
    mutation_input = '\n            entityName: $entity,\n            projectName: $project,\n            runQueues: $queues,\n            hostname: $hostname\n        '
    if 'agentConfig' in self.create_launch_agent_fields_introspection():
        variable_values['agentConfig'] = json.dumps(agent_config)
        mutation_params += ', $agentConfig: JSONString'
        mutation_input += ', agentConfig: $agentConfig'
    if 'version' in self.create_launch_agent_fields_introspection():
        variable_values['version'] = version
        mutation_params += ', $version: String'
        mutation_input += ', version: $version'
    mutation = gql(f'\n            mutation createLaunchAgent(\n                {mutation_params}\n            ) {{\n                createLaunchAgent(\n                    input: {{\n                        {mutation_input}\n                    }}\n                ) {{\n                    launchAgentId\n                }}\n            }}\n            ')
    result: dict = self.gql(mutation, variable_values)['createLaunchAgent']
    return result