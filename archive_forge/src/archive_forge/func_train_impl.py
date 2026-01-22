import functools
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def train_impl(_log_models, _log_datasets, original, *args, **kwargs):

    def record_eval_results(eval_results, metrics_logger):
        """
            Create a callback function that records evaluation results.
            """
        return picklable_exception_safe_function(functools.partial(_autolog_callback, metrics_logger=metrics_logger, eval_results=eval_results))

    def log_feature_importance_plot(features, importance, importance_type):
        """
            Log feature importance plot.
            """
        import matplotlib.pyplot as plt
        indices = np.argsort(importance)
        features = np.array(features)[indices]
        importance = importance[indices]
        num_features = len(features)
        w, h = [6.4, 4.8]
        h = h + 0.1 * num_features if num_features > 10 else h
        fig, ax = plt.subplots(figsize=(w, h))
        yloc = np.arange(num_features)
        ax.barh(yloc, importance, align='center', height=0.5)
        ax.set_yticks(yloc)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance ({importance_type})')
        fig.tight_layout()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                filepath = os.path.join(tmpdir, f'feature_importance_{imp_type}.png')
                fig.savefig(filepath)
                mlflow.log_artifact(filepath)
            finally:
                plt.close(fig)
    autologging_client = MlflowAutologgingQueueingClient()
    booster_params = args[0] if len(args) > 0 else kwargs['params']
    autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=booster_params)
    unlogged_params = ['params', 'train_set', 'valid_sets', 'valid_names', 'fobj', 'feval', 'init_model', 'learning_rates', 'callbacks']
    if Version(lightgbm.__version__) <= Version('3.3.1'):
        unlogged_params.append('evals_result')
    params_to_log_for_fn = get_mlflow_run_params_for_fn_args(original, args, kwargs, unlogged_params)
    autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=params_to_log_for_fn)
    param_logging_operations = autologging_client.flush(synchronous=False)
    all_arg_names = _get_arg_names(original)
    num_pos_args = len(args)
    eval_results = []
    callbacks_index = all_arg_names.index('callbacks')
    run_id = mlflow.active_run().info.run_id
    train_set = args[1] if len(args) > 1 else kwargs.get('train_set')
    if _log_datasets and train_set:
        try:
            context_tags = context_registry.resolve_tags()
            source = CodeDatasetSource(tags=context_tags)
            _log_lightgbm_dataset(train_set, source, 'train', autologging_client)
            valid_sets = kwargs.get('valid_sets')
            if valid_sets is not None:
                valid_names = kwargs.get('valid_names')
                if valid_names is None:
                    for valid_set in valid_sets:
                        _log_lightgbm_dataset(valid_set, source, 'eval', autologging_client)
                else:
                    for valid_set, valid_name in zip(valid_sets, valid_names):
                        _log_lightgbm_dataset(valid_set, source, 'eval', autologging_client, name=valid_name)
            dataset_logging_operations = autologging_client.flush(synchronous=False)
            dataset_logging_operations.await_completion()
        except Exception as e:
            _logger.warning('Failed to log dataset information to MLflow Tracking. Reason: %s', e)
    with batch_metrics_logger(run_id) as metrics_logger:
        callback = record_eval_results(eval_results, metrics_logger)
        if num_pos_args >= callbacks_index + 1:
            tmp_list = list(args)
            tmp_list[callbacks_index] += [callback]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs and kwargs['callbacks'] is not None:
            kwargs['callbacks'] += [callback]
        else:
            kwargs['callbacks'] = [callback]
        model = original(*args, **kwargs)
        early_stopping = model.best_iteration > 0
        if early_stopping:
            extra_step = len(eval_results)
            autologging_client.log_metrics(run_id=mlflow.active_run().info.run_id, metrics={'stopped_iteration': extra_step, 'best_iteration': model.best_iteration})
            last_iter_results = eval_results[model.best_iteration - 1]
            autologging_client.log_metrics(run_id=mlflow.active_run().info.run_id, metrics=last_iter_results, step=extra_step)
            early_stopping_logging_operations = autologging_client.flush(synchronous=False)
    for imp_type in ['split', 'gain']:
        features = model.feature_name()
        importance = model.feature_importance(importance_type=imp_type)
        try:
            log_feature_importance_plot(features, importance, imp_type)
        except Exception:
            _logger.exception('Failed to log feature importance plot. LightGBM autologging will ignore the failure and continue. Exception: ')
        imp = dict(zip(features, importance.tolist()))
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f'feature_importance_{imp_type}.json')
            with open(filepath, 'w') as f:
                json.dump(imp, f, indent=2)
            mlflow.log_artifact(filepath)
    input_example_info = getattr(train_set, 'input_example_info', None)

    def get_input_example():
        if input_example_info is None:
            raise Exception(ENSURE_AUTOLOGGING_ENABLED_TEXT)
        if input_example_info.error_msg is not None:
            raise Exception(input_example_info.error_msg)
        return input_example_info.input_example

    def infer_model_signature(input_example):
        model_output = model.predict(input_example)
        return infer_signature(input_example, model_output)
    if _log_models:
        input_example, signature = resolve_input_example_and_signature(get_input_example, infer_model_signature, log_input_examples, log_model_signatures, _logger)
        log_model(model, artifact_path='model', signature=signature, input_example=input_example, registered_model_name=registered_model_name)
    param_logging_operations.await_completion()
    if early_stopping:
        early_stopping_logging_operations.await_completion()
    return model