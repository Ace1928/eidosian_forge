import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
def dump_input_data(data, inputs_key='inputs', params: Optional[Dict[str, Any]]=None):
    """
    Args:
        data: Input data.
        inputs_key: Key to represent data in the request payload.
        params: Additional parameters to pass to the model for inference.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """
    import numpy as np
    import pandas as pd
    if importlib.util.find_spec('scipy.sparse'):
        from scipy.sparse import csc_matrix, csr_matrix
        if isinstance(data, (csc_matrix, csr_matrix)):
            data = data.toarray()
    if isinstance(data, pd.DataFrame):
        post_data = {'dataframe_split': data.to_dict(orient='split')}
    elif isinstance(data, dict):
        post_data = {inputs_key: {k: get_jsonable_input(k, v) for k, v in data}}
    elif isinstance(data, np.ndarray):
        post_data = {inputs_key: data.tolist()}
    elif isinstance(data, list):
        post_data = {inputs_key: data}
    else:
        post_data = data
    if params is not None:
        if not isinstance(params, dict):
            raise MlflowException(f"Params must be a dictionary. Got type '{type(params).__name__}'.")
        if isinstance(post_data, dict):
            post_data['params'] = params
    if not isinstance(post_data, str):
        post_data = json.dumps(post_data, cls=_CustomJsonEncoder)
    return post_data