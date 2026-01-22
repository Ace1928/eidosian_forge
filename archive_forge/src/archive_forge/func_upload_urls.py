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
def upload_urls(self, project: str, files: Union[List[str], Dict[str, IO]], run: Optional[str]=None, entity: Optional[str]=None, description: Optional[str]=None) -> Tuple[str, List[str], Dict[str, Dict[str, Any]]]:
    """Generate temporary resumable upload urls.

        Arguments:
            project (str): The project to download
            files (list or dict): The filenames to upload
            run (str, optional): The run to upload to
            entity (str, optional): The entity to scope this project to.
            description (str, optional): description

        Returns:
            (run_id, upload_headers, file_info)
            run_id: id of run we uploaded files to
            upload_headers: A list of headers to use when uploading files.
            file_info: A dict of filenames and urls.
                {
                    "run_id": "run_id",
                    "upload_headers": [""],
                    "file_info":  [
                        { "weights.h5": { "uploadUrl": "https://weights.url" } },
                        { "model.json": { "uploadUrl": "https://model.json" } }
                    ]
                }
        """
    run_name = run or self.current_run_id
    assert run_name, 'run must be specified'
    entity = entity or self.settings('entity')
    assert entity, 'entity must be specified'
    has_create_run_files_mutation = self.create_run_files_introspection()
    if not has_create_run_files_mutation:
        return self.legacy_upload_urls(project, files, run, entity, description)
    query = gql('\n        mutation CreateRunFiles($entity: String!, $project: String!, $run: String!, $files: [String!]!) {\n            createRunFiles(input: {entityName: $entity, projectName: $project, runName: $run, files: $files}) {\n                runID\n                uploadHeaders\n                files {\n                    name\n                    uploadUrl\n                }\n            }\n        }\n        ')
    query_result = self.gql(query, variable_values={'project': project, 'run': run_name, 'entity': entity, 'files': [file for file in files]})
    result = query_result['createRunFiles']
    run_id = result['runID']
    if not run_id:
        raise CommError(f'Error uploading files to {entity}/{project}/{run_name}. Check that this project exists and you have access to this entity and project')
    file_name_urls = {file['name']: file for file in result['files']}
    return (run_id, result['uploadHeaders'], file_name_urls)