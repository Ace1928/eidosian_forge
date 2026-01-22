from typing import List, Optional
from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import make_genai_metric
from mlflow.metrics.genai.utils import _get_latest_metric_version
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string

    This function will create a genai metric used to evaluate the evaluate the relevance of an
    LLM using the model provided. Relevance will be assessed by the appropriateness, significance,
    and applicability of the output with respect to the input and ``context``.

    The ``context`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of an openai or gateway judge model in the format of
            "openai:/gpt-4" or "gateway:/my-route". Defaults to
            "openai:/gpt-4". Your use of a third party LLM service (e.g., OpenAI) for
            evaluation may be subject to and governed by the LLM service's terms of use.
        metric_version: (Optional) The version of the relevance metric to use.
            Defaults to the latest version.
        examples: (Optional) Provide a list of examples to help the judge model evaluate the
            relevance. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.

    Returns:
        A metric object
    