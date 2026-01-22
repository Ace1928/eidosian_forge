from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def table_(table: Identifier | str, db: t.Optional[Identifier | str]=None, catalog: t.Optional[Identifier | str]=None, quoted: t.Optional[bool]=None, alias: t.Optional[Identifier | str]=None) -> Table:
    """Build a Table.

    Args:
        table: Table name.
        db: Database name.
        catalog: Catalog name.
        quote: Whether to force quotes on the table's identifiers.
        alias: Table's alias.

    Returns:
        The new Table instance.
    """
    return Table(this=to_identifier(table, quoted=quoted) if table else None, db=to_identifier(db, quoted=quoted) if db else None, catalog=to_identifier(catalog, quoted=quoted) if catalog else None, alias=TableAlias(this=to_identifier(alias)) if alias else None)