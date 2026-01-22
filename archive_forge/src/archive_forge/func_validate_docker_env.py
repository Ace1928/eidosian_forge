import logging
import os
import posixpath
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import docker
from mlflow import tracking
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects.utils import MLFLOW_DOCKER_WORKDIR_PATH
from mlflow.utils import file_utils, process
from mlflow.utils.databricks_utils import get_databricks_env_vars
from mlflow.utils.file_utils import _handle_readonly_on_windows
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_ID, MLFLOW_DOCKER_IMAGE_URI
def validate_docker_env(project):
    if not project.name:
        raise ExecutionException('Project name in MLProject must be specified when using docker for image tagging.')
    if not project.docker_env.get('image'):
        raise ExecutionException("Project with docker environment must specify the docker image to use via an 'image' field under the 'docker_env' field.")