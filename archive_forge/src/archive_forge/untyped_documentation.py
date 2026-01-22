import operator
from decimal import Decimal
from typing import Any, Optional, Tuple, Union
from ..helpers import BOOLEAN_VALUES, get_double
from .atomic_types import AnyAtomicType

        Returns a couple of operands, applying a cast to the instance value based on
        the type of the *other* argument.

        :param other: The other operand, that determines the cast for the untyped instance.
        :param force_float: Force a conversion to float if *other* is an UntypedAtomic instance.
        :return: A couple of values.
        