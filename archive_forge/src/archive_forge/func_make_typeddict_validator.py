import math
import re
from collections import OrderedDict, deque
from collections.abc import Hashable as CollectionsHashable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from . import errors
from .datetime_parse import parse_date, parse_datetime, parse_duration, parse_time
from .typing import (
from .utils import almost_equal_floats, lenient_issubclass, sequence_like
def make_typeddict_validator(typeddict_cls: Type['TypedDict'], config: Type['BaseConfig']) -> Callable[[Any], Dict[str, Any]]:
    from .annotated_types import create_model_from_typeddict
    TypedDictModel = create_model_from_typeddict(typeddict_cls, __config__=config, __module__=typeddict_cls.__module__)
    typeddict_cls.__pydantic_model__ = TypedDictModel

    def typeddict_validator(values: 'TypedDict') -> Dict[str, Any]:
        return TypedDictModel.parse_obj(values).dict(exclude_unset=True)
    return typeddict_validator