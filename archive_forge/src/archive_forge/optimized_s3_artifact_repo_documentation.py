import json
import logging
import math
import os
import posixpath
import urllib.parse
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status

        Instead of returning ArtifactCredentialInfo objects, we instead return a list of initialized
        S3 client. We do so because S3 clients cannot be instantiated within each thread.
        