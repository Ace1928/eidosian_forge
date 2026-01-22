import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
def get_model_dependencies(model_uri, format='pip'):
    """
    Downloads the model dependencies and returns the path to requirements.txt or conda.yaml file.

    .. warning::
        This API downloads all the model artifacts to the local filesystem. This may take
        a long time for large models. To avoid this overhead, use
        ``mlflow.artifacts.download_artifacts("<model_uri>/requirements.txt")`` or
        ``mlflow.artifacts.download_artifacts("<model_uri>/conda.yaml")`` instead.

    Args:
        model_uri: The uri of the model to get dependencies from.
        format: The format of the returned dependency file. If the ``"pip"`` format is
            specified, the path to a pip ``requirements.txt`` file is returned.
            If the ``"conda"`` format is specified, the path to a ``"conda.yaml"``
            file is returned . If the ``"pip"`` format is specified but the model
            was not saved with a ``requirements.txt`` file, the ``pip`` section
            of the model's ``conda.yaml`` file is extracted instead, and any
            additional conda dependencies are ignored. Default value is ``"pip"``.

    Returns:
        The local filesystem path to either a pip ``requirements.txt`` file
        (if ``format="pip"``) or a ``conda.yaml`` file (if ``format="conda"``)
        specifying the model's dependencies.
    """
    dep_file = _get_model_dependencies(model_uri, format)
    if format == 'pip':
        prefix = '%' if _is_in_ipython_notebook() else ''
        _logger.info(f"To install the dependencies that were used to train the model, run the following command: '{prefix}pip install -r {dep_file}'.")
    return dep_file