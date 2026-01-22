import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def cirq_type_from_json(type_str: str, resolvers: Optional[Sequence[JsonResolver]]=None) -> Type:
    """Returns a type object for JSON deserialization of `type_str`.

    This method is not part of the base deserialization path. Together with
    `json_cirq_type`, it can be used to provide type-object deserialization
    for classes that need it.

    Args:
        type_str: string representation of the type to deserialize.
        resolvers: list of JsonResolvers to use in type resolution. If this is
            left blank, DEFAULT_RESOLVERS will be used.

    Returns:
        The type object T for which json_cirq_type(T) matches `type_str`.

    Raises:
        ValueError: if type_str does not have a match in `resolvers`, or if the
            match found is a factory method instead of a type.
    """
    cirq_type = factory_from_json(type_str, resolvers)
    if isinstance(cirq_type, type):
        return cirq_type
    raise ValueError(f'Type {type_str} maps to a factory method instead of a type.')