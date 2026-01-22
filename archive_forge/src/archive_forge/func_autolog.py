import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
@autologging_integration(AUTOLOGGING_INTEGRATION_NAME)
def autolog(log_models=True, log_datasets=True, disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False, log_post_training_metrics=True, registered_model_name=None, log_input_examples=False, log_model_signatures=True, log_model_allowlist=None, extra_tags=None):
    """
    Enables (or disables) and configures autologging for pyspark ml estimators.
    This method is not threadsafe.
    This API requires Spark 3.0 or above.

    **When is autologging performed?**
      Autologging is performed when you call ``Estimator.fit`` except for estimators (featurizers)
      under ``pyspark.ml.feature``.

    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.params``. If a param value is also an ``Estimator``,
          then params in the the wrapped estimator will also be logged, the nested param key
          will be `{estimator_uid}.{param_name}`

      **Tags**
        - An estimator class name (e.g. "LinearRegression").
        - A fully qualified estimator class name
          (e.g. "pyspark.ml.regression.LinearRegression").

      .. _post training metrics:

      **Post training metrics**
        When users call evaluator APIs after model training, MLflow tries to capture the
        `Evaluator.evaluate` results and log them as MLflow metrics to the Run associated with
        the model. All pyspark ML evaluators are supported.

        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"

        - The metric name is the name returned by `Evaluator.getMetricName()`
        - If multiple calls are made to the same pyspark ML evaluator metric, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - MLflow uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the `dataset` argument of `model.transform` call.
          Note: MLflow captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.

        **Limitations**
          - MLflow cannot find run information for other objects derived from a given prediction
            result (e.g. by doing some transformation on the prediction result dataset).

      **Artifacts**
        - An MLflow Model with the :py:mod:`mlflow.spark` flavor containing a fitted estimator
          (logged by :py:func:`mlflow.spark.log_model()`). Note that large models may not be
          autologged for performance and storage space considerations, and autologging for
          Pipelines and hyperparameter tuning meta-estimators (e.g. CrossValidator) is not yet
          supported.
          See ``log_models`` param below for details.
        - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
          JSON object whose keys are MLflow post training metric names
          (see "Post training metrics" section for the key format) and whose values are the
          corresponding evaluator information, including evaluator class name and evaluator params.

    **How does autologging work for meta estimators?**
          When a meta estimator (e.g. `Pipeline`_, `CrossValidator`_, `TrainValidationSplit`_,
          `OneVsRest`_)
          calls ``fit()``, it internally calls ``fit()`` on its child estimators. Autologging
          does NOT perform logging on these constituent ``fit()`` calls.

          A "estimator_info.json" artifact is logged, which includes a `hierarchy` entry
          describing the hierarchy of the meta estimator. The hierarchy includes expanded
          entries for all nested stages, such as nested pipeline stages.

      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`CrossValidator`_ and `TrainValidationSplit`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model and the best parameters (if available).
          For better readability, the "estimatorParamMaps" param in parameter search estimator
          will be recorded inside "estimator_info" artifact, see following description.
          Inside "estimator_info.json" artifact, in addition to the "hierarchy", records 2 more
          items: "tuning_parameter_map_list": a list contains all parameter maps used in tuning,
          and "tuned_estimator_parameter_map": the parameter map of the tuned estimator.
          Records a "best_parameters.json" artifacts, contains the best parameter it searched out.
          Records a "search_results.csv" artifacts, contains search results, it is a table with
          2 columns: "params" and "metric".

    .. _OneVsRest:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.OneVsRest.html#pyspark.ml.classification.OneVsRest
    .. _Pipeline:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html#pyspark.ml.Pipeline
    .. _CrossValidator:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#pyspark.ml.tuning.CrossValidator
    .. _TrainValidationSplit:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.TrainValidationSplit.html#pyspark.ml.tuning.TrainValidationSplit

    Args:
        log_models: If ``True``, if trained models are in allowlist, they are logged as MLflow
            model artifacts. If ``False``, trained models are not logged.
            Note: the built-in allowlist excludes some models (e.g. ALS models) which
            can be large. To specify a custom allowlist, create a file containing a
            newline-delimited list of fully-qualified estimator classnames, and set
            the "spark.mlflow.pysparkml.autolog.logModelAllowlistFile" Spark config
            to the path of your allowlist file.
        log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
            If ``False``, dataset information is not logged.
        disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
            enables the pyspark ML autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            pyspark that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during pyspark ML
            autologging. If ``False``, show all events and warnings during pyspark ML
            autologging.
        log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
            ``True``. See the `post training metrics`_ section for more
            details.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        log_input_examples: If ``True``, input examples from training datasets are collected and
            logged along with pyspark ml model artifacts during training. If
            ``False``, input examples are not logged.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with spark ml pipeline/estimator artifacts during training.
            If ``False`` signatures are not logged.

            .. warning::

                Currently, only scalar Spark data types are supported. If
                model inputs/outputs contain non-scalar Spark data types such
                as ``pyspark.ml.linalg.Vector``, signatures are not logged.

        log_model_allowlist: If given, it overrides the default log model allowlist in mlflow.
            This takes precedence over the spark configuration of
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile".

            **The default log model allowlist in mlflow**
                .. literalinclude:: ../../../mlflow/pyspark/ml/log_model_allowlist.txt
                    :language: text

        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    from pyspark.ml.base import Estimator, Model
    from pyspark.ml.evaluation import Evaluator
    from mlflow.tracking.context import registry as context_registry
    global _log_model_allowlist
    if log_model_allowlist:
        _log_model_allowlist = {model.strip() for model in log_model_allowlist}
    else:
        _log_model_allowlist = _read_log_model_allowlist()

    def _log_pretraining_metadata(estimator, params, input_df):
        if params and isinstance(params, dict):
            estimator = estimator.copy(params)
        autologging_metadata = _gen_estimator_metadata(estimator)
        artifact_dict = {}
        param_map = _get_instance_param_map(estimator, autologging_metadata.uid_to_indexed_name_map)
        if _should_log_hierarchy(estimator):
            artifact_dict['hierarchy'] = autologging_metadata.hierarchy
        for param_search_estimator in autologging_metadata.param_search_estimators:
            param_search_estimator_name = f'{autologging_metadata.uid_to_indexed_name_map[param_search_estimator.uid]}'
            artifact_dict[param_search_estimator_name] = {}
            artifact_dict[param_search_estimator_name]['tuning_parameter_map_list'] = _get_tuning_param_maps(param_search_estimator, autologging_metadata.uid_to_indexed_name_map)
            artifact_dict[param_search_estimator_name]['tuned_estimator_parameter_map'] = _get_instance_param_map_recursively(param_search_estimator.getEstimator(), 1, autologging_metadata.uid_to_indexed_name_map)
        if artifact_dict:
            mlflow.log_dict(artifact_dict, artifact_file='estimator_info.json')
        _log_estimator_params(param_map)
        mlflow.set_tags(_get_estimator_info_tags(estimator))
        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                code_source = CodeDatasetSource(context_tags)
                dataset = SparkDataset(df=input_df, source=code_source)
                mlflow.log_input(dataset, 'train')
            except Exception as e:
                _logger.warning('Failed to log training dataset information to MLflow Tracking. Reason: %s', e)

    def _log_posttraining_metadata(estimator, spark_model, params, input_df):
        if _is_parameter_search_estimator(estimator):
            try:
                child_tags = context_registry.resolve_tags()
                child_tags.update({MLFLOW_AUTOLOGGING: AUTOLOGGING_INTEGRATION_NAME})
                _create_child_runs_for_parameter_search(parent_estimator=estimator, parent_model=spark_model, parent_run=mlflow.active_run(), child_tags=child_tags)
            except Exception:
                msg = f'Encountered exception during creation of child runs for parameter search. Child runs may be missing. Exception: {traceback.format_exc()}'
                _logger.warning(msg)
            estimator_param_maps = _get_tuning_param_maps(estimator, estimator._autologging_metadata.uid_to_indexed_name_map)
            metrics_dict, best_index = _get_param_search_metrics_and_best_index(estimator, spark_model)
            _log_parameter_search_results_as_artifact(estimator_param_maps, metrics_dict, mlflow.active_run().info.run_id)
            best_param_map = estimator_param_maps[best_index]
            mlflow.log_dict(best_param_map, artifact_file='best_parameters.json')
            _log_estimator_params({f'best_{param_name}': param_value for param_name, param_value in best_param_map.items()})
        if log_models:
            if _should_log_model(spark_model):
                from pyspark.sql import SparkSession
                from mlflow.models import infer_signature
                from mlflow.pyspark.ml._autolog import cast_spark_df_with_vector_to_array, get_feature_cols
                from mlflow.spark import _find_and_set_features_col_as_vector_if_needed
                spark = SparkSession.builder.getOrCreate()

                def _get_input_example_as_pd_df():
                    feature_cols = list(get_feature_cols(input_df, spark_model))
                    limited_input_df = input_df.select(feature_cols).limit(INPUT_EXAMPLE_SAMPLE_ROWS)
                    return cast_spark_df_with_vector_to_array(limited_input_df).toPandas()

                def _infer_model_signature(input_example_slice):
                    input_slice_df = _find_and_set_features_col_as_vector_if_needed(spark.createDataFrame(input_example_slice), spark_model)
                    model_output = spark_model.transform(input_slice_df).drop(*input_slice_df.columns)
                    unsupported_columns = _get_columns_with_unsupported_data_type(model_output)
                    if unsupported_columns:
                        _logger.warning(f'Model outputs contain unsupported Spark data types: {unsupported_columns}. Output schema is not be logged.')
                        model_output = None
                    else:
                        model_output = model_output.toPandas()
                    return infer_signature(input_example_slice, model_output)
                nonlocal log_model_signatures
                if log_model_signatures:
                    unsupported_columns = _get_columns_with_unsupported_data_type(input_df)
                    if unsupported_columns:
                        _logger.warning(f'Model inputs contain unsupported Spark data types: {unsupported_columns}. Model signature is not logged.')
                        log_model_signatures = False
                input_example, signature = resolve_input_example_and_signature(_get_input_example_as_pd_df, _infer_model_signature, log_input_examples, log_model_signatures, _logger)
                mlflow.spark.log_model(spark_model, artifact_path='model', registered_model_name=registered_model_name, input_example=input_example, signature=signature)
                if _is_parameter_search_model(spark_model):
                    mlflow.spark.log_model(spark_model.bestModel, artifact_path='best_model')
            else:
                _logger.warning(_get_warning_msg_for_skip_log_model(spark_model))

    def fit_mlflow(original, self, *args, **kwargs):
        params = get_method_call_arg_value(1, 'params', None, args, kwargs)
        if _get_fully_qualified_class_name(self).startswith('pyspark.ml.feature.'):
            return original(self, *args, **kwargs)
        elif isinstance(params, (list, tuple)):
            _logger.warning(_get_warning_msg_for_fit_call_with_a_list_of_params(self))
            return original(self, *args, **kwargs)
        else:
            from pyspark.storagelevel import StorageLevel
            estimator = self.copy(params) if params is not None else self
            input_training_df = args[0].persist(StorageLevel.MEMORY_AND_DISK)
            _log_pretraining_metadata(estimator, params, input_training_df)
            spark_model = original(self, *args, **kwargs)
            _log_posttraining_metadata(estimator, spark_model, params, input_training_df)
            input_training_df.unpersist()
            return spark_model

    def patched_fit(original, self, *args, **kwargs):
        should_log_post_training_metrics = log_post_training_metrics and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        with _SparkTrainingSession(estimator=self, allow_children=False) as t:
            if t.should_log():
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    fit_result = fit_mlflow(original, self, *args, **kwargs)
                if should_log_post_training_metrics and isinstance(fit_result, Model):
                    _AUTOLOGGING_METRICS_MANAGER.register_model(fit_result, mlflow.active_run().info.run_id)
                return fit_result
            else:
                return original(self, *args, **kwargs)

    def patched_transform(original, self, *args, **kwargs):
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            predict_result = original(self, *args, **kwargs)
            eval_dataset = get_method_call_arg_value(0, 'dataset', None, args, kwargs)
            eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(self, eval_dataset)
            _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(run_id, eval_dataset_name, predict_result)
            return predict_result
        else:
            return original(self, *args, **kwargs)

    def patched_evaluate(original, self, *args, **kwargs):
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                metric = original(self, *args, **kwargs)
            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
                params = get_method_call_arg_value(1, 'params', None, args, kwargs)
                evaluator = self.copy(params) if params is not None else self
                metric_name = evaluator.getMetricName()
                evaluator_info = _AUTOLOGGING_METRICS_MANAGER.gen_evaluator_info(evaluator)
                pred_result_dataset = get_method_call_arg_value(0, 'dataset', None, args, kwargs)
                run_id, dataset_name = _AUTOLOGGING_METRICS_MANAGER.get_run_id_and_dataset_name_for_evaluator_call(pred_result_dataset)
                if run_id and dataset_name:
                    metric_key = _AUTOLOGGING_METRICS_MANAGER.register_evaluator_call(run_id, metric_name, dataset_name, evaluator_info)
                    _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(run_id, metric_key, metric)
                    if log_datasets:
                        try:
                            context_tags = context_registry.resolve_tags()
                            code_source = CodeDatasetSource(context_tags)
                            dataset = SparkDataset(df=pred_result_dataset, source=code_source)
                            tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='eval')]
                            dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                            client = MlflowClient()
                            client.log_inputs(run_id, [dataset_input])
                        except Exception as e:
                            _logger.warning('Failed to log evaluation dataset information to MLflow Tracking. Reason: %s', e)
            return metric
        else:
            return original(self, *args, **kwargs)
    safe_patch(AUTOLOGGING_INTEGRATION_NAME, Estimator, 'fit', patched_fit, manage_run=True, extra_tags=extra_tags)
    if log_post_training_metrics:
        safe_patch(AUTOLOGGING_INTEGRATION_NAME, Model, 'transform', patched_transform, manage_run=False)
        safe_patch(AUTOLOGGING_INTEGRATION_NAME, Evaluator, 'evaluate', patched_evaluate, manage_run=False)