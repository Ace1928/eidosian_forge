import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_explainer(explainer, path, serialize_model_using_mlflow=True, conda_env=None, code_paths=None, mlflow_model=None, signature: ModelSignature=None, input_example: ModelInputExample=None, pip_requirements=None, extra_pip_requirements=None, metadata=None):
    """
    Save a SHAP explainer to a path on the local file system. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.shap`
        - :py:mod:`mlflow.pyfunc`

    Args:
        explainer: SHAP explainer to be saved.
        path: Local path where the explainer is to be saved.
        serialize_model_using_mlflow: When set to True, MLflow will extract the underlying
            model and serialize it as an MLmodel, otherwise it uses SHAP's internal serialization.
            Defaults to True. Currently MLflow serialization is only supported for models of
            'sklearn' or 'pytorch' flavors.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>` describes model input
            and output :py:class:`Schema <mlflow.types.Schema>`. The model signature can be
            :py:func:`inferred <mlflow.models.infer_signature>` from datasets with valid model input
            (e.g. the training dataset with target column omitted) and valid model output (e.g.
            model predictions generated on the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """
    import shap
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata
    underlying_model_flavor = None
    underlying_model_path = None
    serializable_by_mlflow = False
    if serialize_model_using_mlflow:
        underlying_model_flavor = get_underlying_model_flavor(explainer.model)
        if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
            serializable_by_mlflow = True
            underlying_model_path = os.path.join(path, _UNDERLYING_MODEL_SUBPATH)
        else:
            warnings.warn('Unable to serialize underlying model using MLflow, will use SHAP serialization')
        if underlying_model_flavor == mlflow.sklearn.FLAVOR_NAME:
            mlflow.sklearn.save_model(explainer.model.inner_model.__self__, underlying_model_path)
        elif underlying_model_flavor == mlflow.pytorch.FLAVOR_NAME:
            mlflow.pytorch.save_model(explainer.model.inner_model, underlying_model_path)
    explainer_data_subpath = 'explainer.shap'
    explainer_output_path = os.path.join(path, explainer_data_subpath)
    with open(explainer_output_path, 'wb') as explainer_output_file_handle:
        if serialize_model_using_mlflow and serializable_by_mlflow:
            explainer.save(explainer_output_file_handle, model_saver=False)
        else:
            explainer.save(explainer_output_file_handle)
    pyfunc.add_to_model(mlflow_model, loader_module='mlflow.shap', model_path=explainer_data_subpath, underlying_model_flavor=underlying_model_flavor, conda_env=_CONDA_ENV_FILE_NAME, python_env=_PYTHON_ENV_FILE_NAME, code=code_dir_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, shap_version=shap.__version__, serialized_explainer=explainer_data_subpath, underlying_model_flavor=underlying_model_flavor, code=code_dir_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(path, FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(default_reqs, pip_requirements, extra_pip_requirements)
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)
    if underlying_model_path is not None:
        underlying_model_conda_env = _get_conda_env_for_underlying_model(underlying_model_path)
        conda_env = _merge_environments(conda_env, underlying_model_conda_env)
        pip_requirements = _get_pip_deps(conda_env)
    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), 'w') as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), '\n'.join(pip_constraints))
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), '\n'.join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))