import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
Creates a new TaggedOperation with combined tags.

        Overloads Operation.with_tags to create a new TaggedOperation
        that has the tags of this operation combined with the new_tags
        specified as the parameter.
        