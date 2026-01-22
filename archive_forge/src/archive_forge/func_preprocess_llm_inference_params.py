from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def preprocess_llm_inference_params(data, flavor_config: Optional[Dict[str, Any]]=None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    When a MLflow inference task is given, return updated `data` and `params` that
    - Extract the parameters from the input data.
    - Replace OpenAI specific parameters with Hugging Face specific parameters, in particular
      - `max_tokens` with `max_new_tokens`
      - `stop` with `stopping_criteria`
    """
    if not isinstance(data, pd.DataFrame):
        raise MlflowException(f'`data` is expected to be a pandas DataFrame for MLflow inference task after signature enforcement, but got type: {type(data)}.')
    updated_data = []
    params = {}
    for column in data.columns:
        if column in ['prompt', 'messages']:
            updated_data = data[column].tolist()
        else:
            param = data[column].tolist()[0]
            if column == 'max_tokens':
                params['max_new_tokens'] = param
            elif column == 'stop':
                params['stopping_criteria'] = _get_stopping_criteria(param, flavor_config.get('source_model_name', None))
            else:
                params[column] = param
    return (updated_data, params)