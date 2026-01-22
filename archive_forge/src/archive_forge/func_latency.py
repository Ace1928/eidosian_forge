from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def latency() -> EvaluationMetric:
    """
    This function will create a metric for calculating latency. Latency is determined by the time
    it takes to generate a prediction for a given input. Note that computing latency requires
    each row to be predicted sequentially, which will likely slow down the evaluation process.
    """
    return make_metric(eval_fn=lambda x: MetricValue(), greater_is_better=False, name='latency')