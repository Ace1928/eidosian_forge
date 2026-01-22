from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import (

        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
        (1) the C++ code necessary to unbox the argument
        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
        :param t: a `Type` of an argument
        :param arg_name: argument name
        :param mutable: boolean for whether this argument type is mutable
        :return: unboxed result
        