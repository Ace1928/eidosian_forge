import datetime
import math
import typing as t
from wandb.util import (
@staticmethod
def types_by_name():
    if TypeRegistry._types_by_name is None:
        TypeRegistry._types_by_name = {}
    return TypeRegistry._types_by_name