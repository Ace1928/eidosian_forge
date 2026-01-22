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
def create_body_model(*, fields: Sequence[ModelField], model_name: str) -> Type[BaseModel]:
    BodyModel = create_model(model_name)
    for f in fields:
        BodyModel.__fields__[f.name] = f
    return BodyModel