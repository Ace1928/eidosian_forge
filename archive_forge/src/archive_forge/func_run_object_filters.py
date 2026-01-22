from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
def run_object_filters(self, object_: SchemaItem, name: sqla_compat._ConstraintName, type_: NameFilterType, reflected: bool, compare_to: Optional[SchemaItem]) -> bool:
    """Run the context's object filters and return True if the targets
        should be part of the autogenerate operation.

        This method should be run for every kind of object encountered within
        an autogenerate operation, giving the environment the chance
        to filter what objects should be included in the comparison.
        The filters here are produced directly via the
        :paramref:`.EnvironmentContext.configure.include_object` parameter.

        """
    for fn in self._object_filters:
        if not fn(object_, name, type_, reflected, compare_to):
            return False
    else:
        return True