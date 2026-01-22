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
def eval_cases(self, baseline_case, compare_cases, eval_metrics, thread_count=-1, eval_step=1):
    """More flexible evaluation of any cases.
            Args:
            baseline_case: Execution case used for baseline
            compare_cases: List of cases to compare
            eval_metrics: Metrics to calculate
            thread_count: thread_count to use.  Will override one in cases
            Returns
            -------
            result : Instance of EvaluationResult class
        """
    if not isinstance(compare_cases, list):
        compare_cases = [compare_cases]
    cases = [baseline_case]
    cases += compare_cases
    for case in cases:
        case._set_thread_count(thread_count)
    metric_result = self._calculate_result_metrics(cases, eval_metrics, thread_count=thread_count, evaluation_step=eval_step)
    metric_result.set_baseline_case(baseline_case)
    return metric_result