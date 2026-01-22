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
def update_artifact_manifest(self, artifact_manifest_id: str, base_artifact_id: Optional[str]=None, digest: Optional[str]=None, include_upload: Optional[bool]=True) -> Tuple[str, Dict[str, Any]]:
    mutation = gql('\n        mutation UpdateArtifactManifest(\n            $artifactManifestID: ID!,\n            $digest: String,\n            $baseArtifactID: ID,\n            $includeUpload: Boolean!,\n        ) {\n            updateArtifactManifest(input: {\n                artifactManifestID: $artifactManifestID,\n                digest: $digest,\n                baseArtifactID: $baseArtifactID,\n            }) {\n                artifactManifest {\n                    id\n                    file {\n                        id\n                        name\n                        displayName\n                        uploadUrl @include(if: $includeUpload)\n                        uploadHeaders @include(if: $includeUpload)\n                    }\n                }\n            }\n        }\n        ')
    response = self.gql(mutation, variable_values={'artifactManifestID': artifact_manifest_id, 'digest': digest, 'baseArtifactID': base_artifact_id, 'includeUpload': include_upload})
    return (response['updateArtifactManifest']['artifactManifest']['id'], response['updateArtifactManifest']['artifactManifest']['file'])