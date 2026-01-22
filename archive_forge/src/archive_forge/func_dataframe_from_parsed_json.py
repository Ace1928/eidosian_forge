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
def dataframe_from_parsed_json(decoded_input, pandas_orient, schema=None):
    """Convert parsed json into pandas.DataFrame. If schema is provided this methods will attempt to
    cast data types according to the schema. This include base64 decoding for binary columns.

    Args:
        decoded_input: Parsed json - either a list or a dictionary.
        schema: MLflow schema used when parsing the data.
        pandas_orient: pandas data frame convention used to store the data.

    Returns:
        pandas.DataFrame.
    """
    import pandas as pd
    if pandas_orient == 'records':
        if not isinstance(decoded_input, list):
            if isinstance(decoded_input, dict):
                typemessage = 'dictionary'
            else:
                typemessage = f'type {type(decoded_input)}'
            raise MlflowInvalidInputException(f'Dataframe records format must be a list of records. Got {typemessage}.')
        try:
            pdf = pd.DataFrame(data=decoded_input)
        except Exception as ex:
            raise MlflowInvalidInputException(f"Provided dataframe_records field is not a valid dataframe representation in 'records' format. Error: '{ex}'")
    elif pandas_orient == 'split':
        if not isinstance(decoded_input, dict):
            if isinstance(decoded_input, list):
                typemessage = 'list'
            else:
                typemessage = f'type {type(decoded_input)}'
            raise MlflowInvalidInputException(f'Dataframe split format must be a dictionary. Got {typemessage}.')
        keys = set(decoded_input.keys())
        missing_data = 'data' not in keys
        extra_keys = keys.difference({'columns', 'data', 'index'})
        if missing_data or extra_keys:
            raise MlflowInvalidInputException(f"Dataframe split format must have 'data' field and optionally 'columns' and 'index' fields. Got {keys}.'")
        try:
            pdf = pd.DataFrame(index=decoded_input.get('index'), columns=decoded_input.get('columns'), data=decoded_input['data'])
        except Exception as ex:
            raise MlflowInvalidInputException(f"Provided dataframe_split field is not a valid dataframe representation in 'split' format. Error: '{ex}'")
    if schema is not None:
        pdf = cast_df_types_according_to_schema(pdf, schema)
    return pdf