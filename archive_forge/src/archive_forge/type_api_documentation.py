from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
evaluate the return type of <self> <op> <othertype>,
            and apply any adaptations to the given operator.

            This method determines the type of a resulting binary expression
            given two source types and an operator.   For example, two
            :class:`_schema.Column` objects, both of the type
            :class:`.Integer`, will
            produce a :class:`.BinaryExpression` that also has the type
            :class:`.Integer` when compared via the addition (``+``) operator.
            However, using the addition operator with an :class:`.Integer`
            and a :class:`.Date` object will produce a :class:`.Date`, assuming
            "days delta" behavior by the database (in reality, most databases
            other than PostgreSQL don't accept this particular operation).

            The method returns a tuple of the form <operator>, <type>.
            The resulting operator and type will be those applied to the
            resulting :class:`.BinaryExpression` as the final operator and the
            right-hand side of the expression.

            Note that only a subset of operators make usage of
            :meth:`._adapt_expression`,
            including math operators and user-defined operators, but not
            boolean comparison or special SQL keywords like MATCH or BETWEEN.

            