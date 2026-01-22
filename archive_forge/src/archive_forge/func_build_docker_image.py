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
def build_docker_image(work_dir, repository_uri, base_image, run_id, build_image, docker_auth):
    """
    Build a docker image containing the project in `work_dir`, using the base image.
    """
    image_uri = _get_docker_image_uri(repository_uri=repository_uri, work_dir=work_dir)
    client = docker.from_env()
    if docker_auth is not None:
        client.login(**docker_auth)
    if not build_image:
        if not client.images.list(name=base_image):
            _logger.info(f'Pulling {base_image}')
            image = client.images.pull(base_image)
        else:
            _logger.info(f'{base_image} already exists')
            image = client.images.get(base_image)
        image_uri = base_image
    else:
        dockerfile = f'FROM {base_image}\n COPY {_PROJECT_TAR_ARCHIVE_NAME}/ {MLFLOW_DOCKER_WORKDIR_PATH}\n WORKDIR {MLFLOW_DOCKER_WORKDIR_PATH}\n'
        build_ctx_path = _create_docker_build_ctx(work_dir, dockerfile)
        with open(build_ctx_path, 'rb') as docker_build_ctx:
            _logger.info('=== Building docker image %s ===', image_uri)
            image, _ = client.images.build(tag=image_uri, forcerm=True, dockerfile=posixpath.join(_PROJECT_TAR_ARCHIVE_NAME, _GENERATED_DOCKERFILE_NAME), fileobj=docker_build_ctx, custom_context=True, encoding='gzip')
        try:
            os.remove(build_ctx_path)
        except Exception:
            _logger.info('Temporary docker context file %s was not deleted.', build_ctx_path)
    tracking.MlflowClient().set_tag(run_id, MLFLOW_DOCKER_IMAGE_URI, image_uri)
    tracking.MlflowClient().set_tag(run_id, MLFLOW_DOCKER_IMAGE_ID, image.id)
    return image