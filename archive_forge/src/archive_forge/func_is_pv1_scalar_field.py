from collections import deque
from copy import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import (
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel, create_model
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin
def is_pv1_scalar_field(field: ModelField) -> bool:
    from fastapi import params
    field_info = field.field_info
    if not (field.shape == SHAPE_SINGLETON and (not lenient_issubclass(field.type_, BaseModel)) and (not lenient_issubclass(field.type_, dict)) and (not field_annotation_is_sequence(field.type_)) and (not is_dataclass(field.type_)) and (not isinstance(field_info, params.Body))):
        return False
    if field.sub_fields:
        if not all((is_pv1_scalar_field(f) for f in field.sub_fields)):
            return False
    return True