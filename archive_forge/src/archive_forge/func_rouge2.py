from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def rouge2() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rouge2`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rouge2`_ uses bigram based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rouge2: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(eval_fn=_rouge2_eval_fn, greater_is_better=True, name='rouge2', version='v1')