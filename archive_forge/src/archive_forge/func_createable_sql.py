from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def createable_sql(self, expression: exp.Create, locations: t.DefaultDict) -> str:
    sql = self.sql(expression, 'this')
    properties = expression.args.get('properties')
    if sql[:1] != '#' and any((isinstance(prop, exp.TemporaryProperty) for prop in (properties.expressions if properties else []))):
        sql = f'#{sql}'
    return sql