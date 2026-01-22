from __future__ import print_function
import os
from copy import copy
from enum import Enum
from .. import CatBoostError, CatBoost
from .evaluation_result import EvaluationResults, MetricEvaluationResult
from ._fold_models_handler import FoldModelsHandler
from ._readers import _SimpleStreamingFileReader
from ._splitter import _Splitter
from .execution_case import ExecutionCase
from .factor_utils import LabelMode, FactorUtils
def eval_features(self, learn_config, features_to_eval, loss_function=None, eval_type=EvalType.SeqAdd, eval_metrics=None, thread_count=-1, eval_step=None, label_mode=LabelMode.AddFeature):
    """ Evaluate features.
            Args:
            learn_config: dict with params or instance of CatBoost. In second case instance params will be used
            features_to_eval: list of indices of features to evaluate
            loss_function: one of CatBoost loss functions, get it from learn_config if not specified
            eval_type: Type of feature evaluate (All, SeqAdd, SeqRem)
            eval_metrics: Additional metrics to calculate
            thread_count: thread_count to use. If not none will override learn_config values
            Returns
            -------
            result : Instance of EvaluationResult class
        """
    features_to_eval = set(features_to_eval)
    if eval_metrics is None:
        eval_metrics = []
    eval_metrics = eval_metrics if isinstance(eval_metrics, list) else [eval_metrics]
    if isinstance(learn_config, CatBoost):
        params = learn_config.get_params()
    else:
        params = dict(learn_config)
    if loss_function is not None:
        if 'loss_function' in params and params['loss_function'] != loss_function:
            raise CatBoostError('Loss function in params {} should be equal to feature evaluation objective function {}'.format(params['loss_function'], loss_function))
    elif 'loss_function' not in params:
        raise CatBoostError('Provide loss function in params or as option to eval_features method')
    if thread_count is not None and thread_count != -1:
        params['thread_count'] = thread_count
    if eval_step is None:
        eval_step = 1
    if loss_function is not None:
        params['loss_function'] = loss_function
    else:
        loss_function = params['loss_function']
    if params['loss_function'] == 'PairLogit':
        raise CatBoostError('Pair classification is not supported')
    baseline_case, test_cases = self._create_eval_feature_cases(params, features_to_eval, eval_type=eval_type, label_mode=label_mode)
    if loss_function not in eval_metrics:
        eval_metrics.append(loss_function)
    return self.eval_cases(baseline_case=baseline_case, compare_cases=test_cases, eval_metrics=eval_metrics, thread_count=thread_count, eval_step=eval_step)