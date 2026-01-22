from typing import List, Optional
from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import make_genai_metric
from mlflow.metrics.genai.utils import _get_latest_metric_version
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
@experimental
def faithfulness(model: Optional[str]=None, metric_version: Optional[str]=_get_latest_metric_version(), examples: Optional[List[EvaluationExample]]=None) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the faithfullness of an LLM using the
    model provided. Faithfulness will be assessed based on how factually consistent the output
    is to the ``context``.

    The ``context`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: Model uri of an openai or gateway judge model in the format of
            "openai:/gpt-4" or "gateway:/my-route". Defaults to
            "openai:/gpt-4". Your use of a third party LLM service (e.g., OpenAI) for
            evaluation may be subject to and governed by the LLM service's terms of use.
        metric_version: The version of the faithfulness metric to use.
            Defaults to the latest version.
        examples: Provide a list of examples to help the judge model evaluate the
            faithfulness. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.

    Returns:
        A metric object
    """
    class_name = f'mlflow.metrics.genai.prompts.{metric_version}.FaithfulnessMetric'
    try:
        faithfulness_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(f'Failed to find faithfulness metric for version {metric_version}. Please check the version', error_code=INVALID_PARAMETER_VALUE) from None
    except Exception as e:
        raise MlflowException(f'Failed to construct faithfulness metric {metric_version}. Error: {e!r}', error_code=INTERNAL_ERROR) from None
    if examples is None:
        examples = faithfulness_class_module.default_examples
    if model is None:
        model = faithfulness_class_module.default_model
    return make_genai_metric(name='faithfulness', definition=faithfulness_class_module.definition, grading_prompt=faithfulness_class_module.grading_prompt, include_input=False, examples=examples, version=metric_version, model=model, grading_context_columns=faithfulness_class_module.grading_context_columns, parameters=faithfulness_class_module.parameters, aggregations=['mean', 'variance', 'p90'], greater_is_better=True)