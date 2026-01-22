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
def group_sql(self, expression: exp.Group) -> str:
    group_by_all = expression.args.get('all')
    if group_by_all is True:
        modifier = ' ALL'
    elif group_by_all is False:
        modifier = ' DISTINCT'
    else:
        modifier = ''
    group_by = self.op_expressions(f'GROUP BY{modifier}', expression)
    grouping_sets = self.expressions(expression, key='grouping_sets', indent=False)
    grouping_sets = f'{self.seg('GROUPING SETS')} {self.wrap(grouping_sets)}' if grouping_sets else ''
    cube = expression.args.get('cube', [])
    if seq_get(cube, 0) is True:
        return f'{group_by}{self.seg('WITH CUBE')}'
    else:
        cube_sql = self.expressions(expression, key='cube', indent=False)
        cube_sql = f'{self.seg('CUBE')} {self.wrap(cube_sql)}' if cube_sql else ''
    rollup = expression.args.get('rollup', [])
    if seq_get(rollup, 0) is True:
        return f'{group_by}{self.seg('WITH ROLLUP')}'
    else:
        rollup_sql = self.expressions(expression, key='rollup', indent=False)
        rollup_sql = f'{self.seg('ROLLUP')} {self.wrap(rollup_sql)}' if rollup_sql else ''
    groupings = csv(grouping_sets, cube_sql, rollup_sql, self.seg('WITH TOTALS') if expression.args.get('totals') else '', sep=self.GROUPINGS_SEP)
    if expression.args.get('expressions') and groupings:
        group_by = f'{group_by}{self.GROUPINGS_SEP}'
    return f'{group_by}{groupings}'