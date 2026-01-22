import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from typing_extensions import Annotated, Literal
from .fields import (
from .json import pydantic_encoder
from .networks import AnyUrl, EmailStr
from .types import (
from .typing import (
from .utils import ROOT_KEY, get_model, lenient_issubclass
def multitypes_literal_field_for_schema(values: Tuple[Any, ...], field: ModelField) -> ModelField:
    """
    To support `Literal` with values of different types, we split it into multiple `Literal` with same type
    e.g. `Literal['qwe', 'asd', 1, 2]` becomes `Union[Literal['qwe', 'asd'], Literal[1, 2]]`
    """
    literal_distinct_types = defaultdict(list)
    for v in values:
        literal_distinct_types[v.__class__].append(v)
    distinct_literals = (Literal[tuple(same_type_values)] for same_type_values in literal_distinct_types.values())
    return ModelField(name=field.name, type_=Union[tuple(distinct_literals)], class_validators=field.class_validators, model_config=field.model_config, default=field.default, required=field.required, alias=field.alias, field_info=field.field_info)