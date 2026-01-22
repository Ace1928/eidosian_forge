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
def comparison_func(column, value):
    if comparator == 'LIKE':
        return column.like(value)
    elif comparator == 'ILIKE':
        return column.ilike(value)
    elif comparator == 'IN':
        return column.in_(value)
    elif comparator == 'NOT IN':
        return ~column.in_(value)
    return SearchUtils.get_comparison_func(comparator)(column, value)