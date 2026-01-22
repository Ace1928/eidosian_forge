import base64
import datetime
import os
import posixpath
import re
import urllib.parse
from typing import Union
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.credentials import get_default_host_creds
@staticmethod
def parse_wasbs_uri(uri):
    """Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix)."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != 'wasbs':
        raise Exception(f'Not a WASBS URI: {uri}')
    match = re.match('([^@]+)@([^.]+)\\.(blob\\.core\\.(windows\\.net|chinacloudapi\\.cn))', parsed.netloc)
    if match is None:
        raise Exception('WASBS URI must be of the form <container>@<account>.blob.core.windows.net or <container>@<account>.blob.core.chinacloudapi.cn')
    container = match.group(1)
    storage_account = match.group(2)
    api_uri_suffix = match.group(3)
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return (container, storage_account, path, api_uri_suffix)