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

    Args:
        data: Input data.
        inputs_key: Key to represent data in the request payload.
        params: Additional parameters to pass to the model for inference.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    