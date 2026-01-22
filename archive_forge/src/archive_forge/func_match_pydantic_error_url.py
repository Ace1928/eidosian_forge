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
def match_pydantic_error_url(error_type: str) -> Any:
    from dirty_equals import IsStr
    return IsStr(regex=f'^https://errors\\.pydantic\\.dev/.*/v/{error_type}')