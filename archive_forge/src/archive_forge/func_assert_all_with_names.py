from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def assert_all_with_names(self) -> 'SelectColumns':
    """Assert every column have explicit alias or the alias can
        be inferred (non empty value). It will also validate there is
        no duplicated aliases

        :raises ValueError: if there are columns without alias, or there are
          duplicated names.
        :return: the instance itself
        """
    names: Set[str] = set()
    for x in self.all_cols:
        if isinstance(x, _WildcardExpr):
            continue
        if isinstance(x, _NamedColumnExpr):
            if self._has_wildcard:
                if x.as_name == '':
                    raise ValueError(f"with '*', all other columns must have an alias: {self}")
        if x.output_name == '':
            raise ValueError(f'{x} does not have an alias: {self}')
        if x.output_name in names:
            raise ValueError(f"{x} can't be reused in select: {self}")
        names.add(x.output_name)
    return self