import re
import warnings
from dataclasses import is_dataclass
from typing import (
from weakref import WeakKeyDictionary
import fastapi
from fastapi._compat import (
from fastapi.datastructures import DefaultPlaceholder, DefaultType
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Literal
def generate_operation_id_for_path(*, name: str, path: str, method: str) -> str:
    warnings.warn('fastapi.utils.generate_operation_id_for_path() was deprecated, it is not used internally, and will be removed soon', DeprecationWarning, stacklevel=2)
    operation_id = f'{name}{path}'
    operation_id = re.sub('\\W', '_', operation_id)
    operation_id = f'{operation_id}_{method.lower()}'
    return operation_id