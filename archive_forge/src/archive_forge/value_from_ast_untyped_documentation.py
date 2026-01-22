from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
Produce a Python value given a GraphQL Value AST.

    Unlike :func:`~graphql.utilities.value_from_ast`, no type is provided.
    The resulting Python value will reflect the provided GraphQL value AST.

    =================== ============== ================
       GraphQL Value      JSON Value     Python Value
    =================== ============== ================
       Input Object       Object         dict
       List               Array          list
       Boolean            Boolean        bool
       String / Enum      String         str
       Int / Float        Number         int / float
       Null               null           None
    =================== ============== ================

    