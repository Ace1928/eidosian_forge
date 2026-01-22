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
@pandas_udf(result_type)
def udf(iterator: Iterator[Tuple[Union[pandas.Series, pandas.DataFrame], ...]]) -> Iterator[result_type_hint]:
    from mlflow.pyfunc.scoring_server.client import ScoringServerClient, StdinScoringServerClient
    update_envs = {}
    if mlflow_home is not None:
        update_envs['MLFLOW_HOME'] = mlflow_home
    if openai_env_vars:
        update_envs.update(openai_env_vars)
    if mlflow_testing:
        update_envs[_MLFLOW_TESTING.name] = mlflow_testing
    if extra_env:
        update_envs.update(extra_env)
    with modified_environ(update=update_envs):
        scoring_server_proc = None
        mlflow.set_tracking_uri(tracking_uri)
        if env_manager != _EnvManager.LOCAL:
            if should_use_spark_to_broadcast_file:
                local_model_path_on_executor = _SparkDirectoryDistributor.get_or_extract(archive_path)
                pyfunc_backend.prepare_env(model_uri=local_model_path_on_executor, capture_output=True)
            else:
                local_model_path_on_executor = None
            if check_port_connectivity():
                server_port = find_free_port()
                host = '127.0.0.1'
                scoring_server_proc = pyfunc_backend.serve(model_uri=local_model_path_on_executor or local_model_path, port=server_port, host=host, timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(), enable_mlserver=False, synchronous=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                client = ScoringServerClient(host, server_port)
            else:
                scoring_server_proc = pyfunc_backend.serve_stdin(model_uri=local_model_path_on_executor or local_model_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                client = StdinScoringServerClient(scoring_server_proc)
            _logger.info('Using %s', client.__class__.__name__)
            server_tail_logs = collections.deque(maxlen=_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP)

            def server_redirect_log_thread_func(child_stdout):
                for line in child_stdout:
                    decoded = line.decode() if isinstance(line, bytes) else line
                    server_tail_logs.append(decoded)
                    sys.stdout.write('[model server] ' + decoded)
            server_redirect_log_thread = threading.Thread(target=server_redirect_log_thread_func, args=(scoring_server_proc.stdout,), daemon=True)
            server_redirect_log_thread.start()
            try:
                client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
            except Exception as e:
                err_msg = 'During spark UDF task execution, mlflow model server failed to launch. '
                if len(server_tail_logs) == _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP:
                    err_msg += f'Last {_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP} lines of MLflow model server output:\n'
                else:
                    err_msg += 'MLflow model server output:\n'
                err_msg += ''.join(server_tail_logs)
                raise MlflowException(err_msg) from e

            def batch_predict_fn(pdf, params=None):
                if inspect.signature(client.invoke).parameters.get('params'):
                    return client.invoke(pdf, params=params).get_predictions()
                _log_warning_if_params_not_in_predict_signature(_logger, params)
                return client.invoke(pdf).get_predictions()
        elif env_manager == _EnvManager.LOCAL:
            if is_spark_connect and (not should_spark_connect_use_nfs):
                model_path = os.path.join(tempfile.gettempdir(), 'mlflow', insecure_hash.sha1(model_uri.encode()).hexdigest())
                try:
                    loaded_model = mlflow.pyfunc.load_model(model_path)
                except Exception:
                    os.makedirs(model_path, exist_ok=True)
                    loaded_model = mlflow.pyfunc.load_model(model_uri, dst_path=model_path)
            elif should_use_spark_to_broadcast_file:
                loaded_model, _ = SparkModelCache.get_or_load(archive_path)
            else:
                loaded_model = mlflow.pyfunc.load_model(local_model_path)

            def batch_predict_fn(pdf, params=None):
                if inspect.signature(loaded_model.predict).parameters.get('params'):
                    return loaded_model.predict(pdf, params=params)
                _log_warning_if_params_not_in_predict_signature(_logger, params)
                return loaded_model.predict(pdf)
        try:
            for input_batch in iterator:
                if isinstance(input_batch, (pandas.Series, pandas.DataFrame)):
                    row_batch_args = (input_batch,)
                else:
                    row_batch_args = input_batch
                if len(row_batch_args[0]) > 0:
                    yield _predict_row_batch(batch_predict_fn, row_batch_args)
        finally:
            if scoring_server_proc is not None:
                os.kill(scoring_server_proc.pid, signal.SIGTERM)