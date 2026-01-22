from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def schema_of_csv(col: ColumnOrName, options: t.Optional[t.Dict[str, str]]=None) -> Column:
    if options is not None:
        options_col = create_map([lit(x) for x in _flatten(options.items())])
        return Column.invoke_anonymous_function(col, 'SCHEMA_OF_CSV', options_col)
    return Column.invoke_anonymous_function(col, 'SCHEMA_OF_CSV')