import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import (
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
def score_model_on_one_payload(payload, eval_model):
    try:
        raw_result = model_utils.score_model_on_payload(eval_model, payload, eval_parameters)
        return _extract_score_and_justification(raw_result)
    except ImportError:
        raise
    except MlflowException as e:
        if e.error_code in [ErrorCode.Name(BAD_REQUEST), ErrorCode.Name(UNAUTHENTICATED), ErrorCode.Name(INVALID_PARAMETER_VALUE)]:
            raise
        else:
            return (None, f'Failed to score model on payload. Error: {e!s}')
    except Exception as e:
        return (None, f'Failed to score model on payload. Error: {e!s}')