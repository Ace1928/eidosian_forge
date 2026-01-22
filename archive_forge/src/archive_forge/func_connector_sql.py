from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def connector_sql(self, expression: exp.Connector, op: str, stack: t.Optional[t.List[str | exp.Expression]]=None) -> str:
    if stack is not None:
        if expression.expressions:
            stack.append(self.expressions(expression, sep=f' {op} '))
        else:
            stack.append(expression.right)
            if expression.comments:
                for comment in expression.comments:
                    op += f' /*{self.pad_comment(comment)}*/'
            stack.extend((op, expression.left))
        return op
    stack = [expression]
    sqls: t.List[str] = []
    ops = set()
    while stack:
        node = stack.pop()
        if isinstance(node, exp.Connector):
            ops.add(getattr(self, f'{node.key}_sql')(node, stack))
        else:
            sql = self.sql(node)
            if sqls and sqls[-1] in ops:
                sqls[-1] += f' {sql}'
            else:
                sqls.append(sql)
    sep = '\n' if self.pretty and self.too_wide(sqls) else ' '
    return sep.join(sqls)