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
def spark_udf(spark, model_uri, result_type=None, env_manager=_EnvManager.LOCAL, params: Optional[Dict[str, Any]]=None, extra_env: Optional[Dict[str, str]]=None):
    """
    A Spark UDF that can be used to invoke the Python function formatted model.

    Parameters passed to the UDF are forwarded to the model as a DataFrame where the column names
    are ordinals (0, 1, ...). On some versions of Spark (3.0 and above), it is also possible to
    wrap the input in a struct. In that case, the data will be passed as a DataFrame with column
    names given by the struct definition (e.g. when invoked as my_udf(struct('x', 'y')), the model
    will get the data as a pandas DataFrame with 2 columns 'x' and 'y').

    If a model contains a signature with tensor spec inputs, you will need to pass a column of
    array type as a corresponding UDF argument. The column values of which must be one dimensional
    arrays. The UDF will reshape the column values to the required shape with 'C' order
    (i.e. read / write the elements using C-like index order) and cast the values as the required
    tensor spec type.

    If a model contains a signature, the UDF can be called without specifying column name
    arguments. In this case, the UDF will be called with column names from signature, so the
    evaluation dataframe's column names must match the model signature's column names.

    The predictions are filtered to contain only the columns that can be represented as the
    ``result_type``. If the ``result_type`` is string or array of strings, all predictions are
    converted to string. If the result type is not an array type, the left most column with
    matching type is returned.

    NOTE: Inputs of type ``pyspark.sql.types.DateType`` are not supported on earlier versions of
    Spark (2.4 and below).

    .. code-block:: python
        :caption: Example

        from pyspark.sql.functions import struct

        predict = mlflow.pyfunc.spark_udf(spark, "/my/local/model")
        df.withColumn("prediction", predict(struct("name", "age"))).show()

    Args:
        spark: A SparkSession object.
        model_uri: The location, in URI format, of the MLflow model with the
            :py:mod:`mlflow.pyfunc` flavor. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

        result_type: the return type of the user-defined function. The value can be either a
            ``pyspark.sql.types.DataType`` object or a DDL-formatted type string. Only a primitive
            type, an array ``pyspark.sql.types.ArrayType`` of primitive type, or a struct type
            containing fields of above 2 kinds of types are allowed.
            If unspecified, it tries to infer result type from model signature
            output schema, if model output schema is not available, it fallbacks to use ``double``
            type.

            The following classes of result type are supported:

            - "int" or ``pyspark.sql.types.IntegerType``: The leftmost integer that can fit in an
              ``int32`` or an exception if there is none.

            - "long" or ``pyspark.sql.types.LongType``: The leftmost long integer that can fit in an
              ``int64`` or an exception if there is none.

            - ``ArrayType(IntegerType|LongType)``: All integer columns that can fit into the
              requested size.

            - "float" or ``pyspark.sql.types.FloatType``: The leftmost numeric result cast to
              ``float32`` or an exception if there is none.

            - "double" or ``pyspark.sql.types.DoubleType``: The leftmost numeric result cast to
              ``double`` or an exception if there is none.

            - ``ArrayType(FloatType|DoubleType)``: All numeric columns cast to the requested type or
              an exception if there are no numeric columns.

            - "string" or ``pyspark.sql.types.StringType``: The leftmost column converted to
              ``string``.

            - "boolean" or "bool" or ``pyspark.sql.types.BooleanType``: The leftmost column
              converted to ``bool`` or an exception if there is none.

            - ``ArrayType(StringType)``: All columns converted to ``string``.

            - "field1 FIELD1_TYPE, field2 FIELD2_TYPE, ...": A struct type containing multiple
              fields separated by comma, each field type must be one of types listed above.

        env_manager: The environment manager to use in order to create the python environment
            for model inference. Note that environment is only restored in the context
            of the PySpark UDF; the software environment outside of the UDF is
            unaffected. Default value is ``local``, and the following values are
            supported:

            - ``virtualenv``: Use virtualenv to restore the python environment that
              was used to train the model.
            - ``conda``: (Recommended) Use Conda to restore the software environment
              that was used to train the model.
            - ``local``: Use the current Python environment for model inference, which
              may differ from the environment used to train the model and may lead to
              errors or invalid predictions.

        params: Additional parameters to pass to the model for inference.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.

        extra_env: Extra environment variables to pass to the UDF executors.

    Returns:
        Spark UDF that applies the model's ``predict`` method to the data and returns a
        type specified by ``result_type``, which by default is a double.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, BooleanType, DoubleType, FloatType, IntegerType, LongType, StringType
    from pyspark.sql.types import StructType as SparkStructType
    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from mlflow.utils._spark_utils import _SparkDirectoryDistributor
    is_spark_connect = _is_spark_connect()
    mlflow_home = os.environ.get('MLFLOW_HOME')
    openai_env_vars = mlflow.openai._OpenAIEnvVar.read_environ()
    mlflow_testing = _MLFLOW_TESTING.get_raw()
    _EnvManager.validate(env_manager)
    is_spark_in_local_mode = spark.conf.get('spark.master').startswith('local')
    nfs_root_dir = get_nfs_cache_root_dir()
    should_use_nfs = nfs_root_dir is not None
    should_use_spark_to_broadcast_file = not (is_spark_in_local_mode or should_use_nfs or is_spark_connect)
    should_spark_connect_use_nfs = is_in_databricks_runtime() and should_use_nfs
    if is_spark_connect and env_manager in (_EnvManager.VIRTUALENV, _EnvManager.CONDA) and (not should_spark_connect_use_nfs):
        raise MlflowException.invalid_parameter_value(f'Environment manager {env_manager!r} is not supported in Spark connect mode when either non-Databricks environment is in use or NFS is unavailable.')
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=_create_model_downloading_tmp_dir(should_use_nfs))
    if env_manager == _EnvManager.LOCAL:
        model_requirements = _get_pip_requirements_from_model_path(local_model_path)
        warn_dependency_requirement_mismatches(model_requirements)
        _logger.warning('Calling `spark_udf()` with `env_manager="local"` does not recreate the same environment that was used during training, which may lead to errors or inaccurate predictions. We recommend specifying `env_manager="conda"`, which automatically recreates the environment that was used to train the model and performs inference in the recreated environment.')
    else:
        _logger.info(f"This UDF will use {env_manager} to recreate the model's software environment for inference. This may take extra time during execution.")
        if not sys.platform.startswith('linux'):
            _logger.warning('In order to run inference code in restored python environment, PySpark UDF processes spawn MLflow Model servers as child processes. Due to system limitations with handling SIGKILL signals, these MLflow Model server child processes cannot be cleaned up if the Spark Job is canceled.')
    pyfunc_backend = get_flavor_backend(local_model_path, env_manager=env_manager, install_mlflow=os.environ.get('MLFLOW_HOME') is not None, create_env_root_dir=True)
    if not should_use_spark_to_broadcast_file:
        if env_manager != _EnvManager.LOCAL:
            pyfunc_backend.prepare_env(model_uri=local_model_path, capture_output=is_in_databricks_runtime())
    else:
        archive_path = SparkModelCache.add_local_model(spark, local_model_path)
    model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))
    if result_type is None:
        if (model_output_schema := model_metadata.get_output_schema()):
            result_type = _infer_spark_udf_return_type(model_output_schema)
        else:
            _logger.warning("No 'result_type' provided for spark_udf and the model does not have an output schema. 'result_type' is set to 'double' type.")
            result_type = DoubleType()
    elif isinstance(result_type, str):
        result_type = _parse_spark_datatype(result_type)
    if not _check_udf_return_type(result_type):
        raise MlflowException.invalid_parameter_value(f"Invalid 'spark_udf' result type: {result_type}.\nIt must be one of the following types:\nPrimitive types:\n - int\n - long\n - float\n - double\n - string\n - boolean\nCompound types:\n - ND array of primitives / structs.\n - struct<field: primitive | array<primitive> | array<array<primitive>>, ...>:\n   A struct with primitive, ND array<primitive/structs>,\n   e.g., struct<a:int, b:array<int>>.\n")
    params = _validate_params(params, model_metadata)

    def _predict_row_batch(predict_fn, args):
        input_schema = model_metadata.get_input_schema()
        args = list(args)
        if len(args) == 1 and isinstance(args[0], pandas.DataFrame):
            pdf = args[0]
        else:
            if input_schema is None:
                names = [str(i) for i in range(len(args))]
            else:
                names = input_schema.input_names()
                required_names = input_schema.required_input_names()
                if len(args) > len(names):
                    args = args[:len(names)]
                if len(args) < len(required_names):
                    raise MlflowException('Model input is missing required columns. Expected {} required input columns {}, but the model received only {} unnamed input columns (Since the columns were passed unnamed they are expected to be in the order specified by the schema).'.format(len(names), names, len(args)))
            pdf = pandas.DataFrame(data={names[i]: arg if isinstance(arg, pandas.Series) else arg.apply(lambda row: row.to_dict(), axis=1) for i, arg in enumerate(args)}, columns=names)
        result = predict_fn(pdf, params)
        if isinstance(result, dict):
            result = {k: list(v) for k, v in result.items()}
        if isinstance(result_type, ArrayType) and isinstance(result_type.elementType, ArrayType):
            result_values = _convert_array_values(result, result_type)
            return pandas.Series(result_values)
        if not isinstance(result, pandas.DataFrame):
            result = pandas.DataFrame([result]) if np.isscalar(result) else pandas.DataFrame(result)
        if isinstance(result_type, SparkStructType):
            return _convert_struct_values(result, result_type)
        elem_type = result_type.elementType if isinstance(result_type, ArrayType) else result_type
        if type(elem_type) == IntegerType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, np.int32]).astype(np.int32)
        elif type(elem_type) == LongType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, int]).astype(np.int64)
        elif type(elem_type) == FloatType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float32)
        elif type(elem_type) == DoubleType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float64)
        elif type(elem_type) == BooleanType:
            result = result.select_dtypes([bool, np.bool_]).astype(bool)
        if len(result.columns) == 0:
            raise MlflowException(message=f"The model did not produce any values compatible with the requested type '{elem_type}'. Consider requesting udf with StringType or Arraytype(StringType).", error_code=INVALID_PARAMETER_VALUE)
        if type(elem_type) == StringType:
            result = result.applymap(str)
        if type(result_type) == ArrayType:
            return pandas.Series(result.to_numpy().tolist())
        else:
            return result[result.columns[0]]
    result_type_hint = pandas.DataFrame if isinstance(result_type, SparkStructType) else pandas.Series
    tracking_uri = mlflow.get_tracking_uri()

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
    udf.metadata = model_metadata

    @functools.wraps(udf)
    def udf_with_default_cols(*args):
        if len(args) == 0:
            input_schema = model_metadata.get_input_schema()
            if input_schema and len(input_schema.optional_input_names()) > 0:
                raise MlflowException(message='Cannot apply UDF without column names specified when model signature contains optional columns.', error_code=INVALID_PARAMETER_VALUE)
            if input_schema and len(input_schema.inputs) > 0:
                if input_schema.has_input_names():
                    input_names = input_schema.input_names()
                    return udf(*input_names)
                else:
                    raise MlflowException(message='Cannot apply udf because no column names specified. The udf expects {} columns with types: {}. Input column names could not be inferred from the model signature (column names not found).'.format(len(input_schema.inputs), input_schema.inputs), error_code=INVALID_PARAMETER_VALUE)
            else:
                raise MlflowException('Attempting to apply udf on zero columns because no column names were specified as arguments or inferred from the model signature.', error_code=INVALID_PARAMETER_VALUE)
        else:
            return udf(*args)
    return udf_with_default_cols