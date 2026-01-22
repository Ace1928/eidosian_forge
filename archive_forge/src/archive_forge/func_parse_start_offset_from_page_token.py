import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
@classmethod
def parse_start_offset_from_page_token(cls, page_token):
    if not page_token:
        return 0
    try:
        decoded_token = base64.b64decode(page_token)
    except TypeError:
        raise MlflowException('Invalid page token, could not base64-decode', error_code=INVALID_PARAMETER_VALUE)
    except base64.binascii.Error:
        raise MlflowException('Invalid page token, could not base64-decode', error_code=INVALID_PARAMETER_VALUE)
    try:
        parsed_token = json.loads(decoded_token)
    except ValueError:
        raise MlflowException(f'Invalid page token, decoded value={decoded_token}', error_code=INVALID_PARAMETER_VALUE)
    offset_str = parsed_token.get('offset')
    if not offset_str:
        raise MlflowException(f'Invalid page token, parsed value={parsed_token}', error_code=INVALID_PARAMETER_VALUE)
    try:
        offset = int(offset_str)
    except ValueError:
        raise MlflowException(f'Invalid page token, not stringable {offset_str}', error_code=INVALID_PARAMETER_VALUE)
    return offset