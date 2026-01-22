import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
def patched_inference(func_name, original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables logging the
    following parameters, metrics and artifacts:

    - model
    - metrics
    - data

    We patch either `invoke` or `__call__` function for different models
    based on their usage.
    """
    import langchain
    from langchain_community.callbacks import MlflowCallbackHandler

    class _MlflowLangchainCallback(MlflowCallbackHandler, metaclass=ExceptionSafeAbstractClass):
        """
        Callback for auto-logging metrics and parameters.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    _lc_version = Version(langchain.__version__)
    if not MIN_REQ_VERSION <= _lc_version <= MAX_REQ_VERSION:
        warnings.warn(f'Autologging is known to be compatible with langchain versions between {MIN_REQ_VERSION} and {MAX_REQ_VERSION} and may not succeed with packages outside this range.')
    run_id = getattr(self, 'run_id', None)
    active_run = mlflow.active_run()
    if run_id is None:
        extra_tags = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'extra_tags', None)
        resolved_tags = context_registry.resolve_tags(extra_tags)
        tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, resolved_tags)
        if active_run:
            run_id = active_run.info.run_id
            mlflow.MlflowClient().log_batch(run_id=run_id, tags=[RunTag(key, str(value)) for key, value in tags.items()])
    else:
        tags = None
    session_id = getattr(self, 'session_id', uuid.uuid4().hex)
    inference_id = getattr(self, 'inference_id', 0)
    mlflow_callback = _MlflowLangchainCallback(tracking_uri=mlflow.get_tracking_uri(), run_id=run_id, artifacts_dir=f'artifacts-{session_id}-{inference_id}', tags=tags)
    args, kwargs = _inject_mlflow_callback(func_name, mlflow_callback, args, kwargs)
    with disable_autologging():
        result = original(self, *args, **kwargs)
    mlflow_callback.flush_tracker()
    log_models = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'log_models', False)
    log_input_examples = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'log_input_examples', False)
    log_model_signatures = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'log_model_signatures', False)
    input_example = None
    if log_models and (not hasattr(self, 'model_logged')):
        if func_name == 'get_relevant_documents' or _runnable_with_retriever(self) or _chain_with_retriever(self):
            _logger.info(UNSUPPORT_LOG_MODEL_MESSAGE)
        else:
            warnings.warn(UNSUPPORT_LOG_MODEL_MESSAGE)
            if log_input_examples:
                input_example = deepcopy(_get_input_data_from_function(func_name, self, args, kwargs))
                if not log_model_signatures:
                    _logger.info('Signature is automatically generated for logged model if input_example is provided. To disable log_model_signatures, please also disable log_input_examples.')
            registered_model_name = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'registered_model_name', None)
            try:
                with disable_autologging():
                    mlflow.langchain.log_model(self, 'model', input_example=input_example, registered_model_name=registered_model_name, run_id=mlflow_callback.mlflg.run_id)
            except Exception as e:
                _logger.warning(f'Failed to log model due to error {e}.')
            if _update_langchain_model_config(self):
                self.model_logged = True
    if _update_langchain_model_config(self):
        if not hasattr(self, 'run_id'):
            self.run_id = mlflow_callback.mlflg.run_id
        if not hasattr(self, 'session_id'):
            self.session_id = session_id
        self.inference_id = inference_id + 1
    log_inputs_outputs = get_autologging_config(mlflow.langchain.FLAVOR_NAME, 'log_inputs_outputs', False)
    if log_inputs_outputs:
        if input_example is None:
            input_data = deepcopy(_get_input_data_from_function(func_name, self, args, kwargs))
            if input_data is None:
                _logger.info('Input data gathering failed, only log inference results.')
        else:
            input_data = input_example
        try:
            data_dict = _combine_input_and_output(input_data, result, self.session_id, func_name)
        except Exception as e:
            _logger.warning(f'Failed to log inputs and outputs into `{INFERENCE_FILE_NAME}` file due to error {e}.')
        mlflow.log_table(data_dict, INFERENCE_FILE_NAME, run_id=mlflow_callback.mlflg.run_id)
    if active_run is None or active_run.info.run_id != mlflow_callback.mlflg.run_id:
        mlflow.MlflowClient().set_terminated(mlflow_callback.mlflg.run_id)
    return result