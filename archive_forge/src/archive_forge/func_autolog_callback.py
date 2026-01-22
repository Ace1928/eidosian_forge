import logging
import xgboost
from packaging.version import Version
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass
def autolog_callback(env, metrics_logger, eval_results):
    metric_dict = _patch_metric_names(dict(env.evaluation_result_list))
    metrics_logger.record_metrics(metric_dict, env.iteration)
    eval_results.append(metric_dict)