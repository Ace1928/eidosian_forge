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
def get_schema_from_model_field(*, field: ModelField, schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[Tuple[ModelField, Literal['validation', 'serialization']], JsonSchemaValue], separate_input_output_schemas: bool=True) -> Dict[str, Any]:
    return field_schema(field, model_name_map=model_name_map, ref_prefix=REF_PREFIX)[0]