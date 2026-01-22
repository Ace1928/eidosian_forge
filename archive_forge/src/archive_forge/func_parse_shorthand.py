from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def parse_shorthand(shorthand: Union[Dict[str, Any], str], data: Optional[Union[pd.DataFrame, DataFrameLike]]=None, parse_aggregates: bool=True, parse_window_ops: bool=False, parse_timeunits: bool=True, parse_types: bool=True) -> Dict[str, Any]:
    """General tool to parse shorthand values

    These are of the form:

    - "col_name"
    - "col_name:O"
    - "average(col_name)"
    - "average(col_name):O"

    Optionally, a dataframe may be supplied, from which the type
    will be inferred if not specified in the shorthand.

    Parameters
    ----------
    shorthand : dict or string
        The shorthand representation to be parsed
    data : DataFrame, optional
        If specified and of type DataFrame, then use these values to infer the
        column type if not provided by the shorthand.
    parse_aggregates : boolean
        If True (default), then parse aggregate functions within the shorthand.
    parse_window_ops : boolean
        If True then parse window operations within the shorthand (default:False)
    parse_timeunits : boolean
        If True (default), then parse timeUnits from within the shorthand
    parse_types : boolean
        If True (default), then parse typecodes within the shorthand

    Returns
    -------
    attrs : dict
        a dictionary of attributes extracted from the shorthand

    Examples
    --------
    >>> data = pd.DataFrame({'foo': ['A', 'B', 'A', 'B'],
    ...                      'bar': [1, 2, 3, 4]})

    >>> parse_shorthand('name') == {'field': 'name'}
    True

    >>> parse_shorthand('name:Q') == {'field': 'name', 'type': 'quantitative'}
    True

    >>> parse_shorthand('average(col)') == {'aggregate': 'average', 'field': 'col'}
    True

    >>> parse_shorthand('foo:O') == {'field': 'foo', 'type': 'ordinal'}
    True

    >>> parse_shorthand('min(foo):Q') == {'aggregate': 'min', 'field': 'foo', 'type': 'quantitative'}
    True

    >>> parse_shorthand('month(col)') == {'field': 'col', 'timeUnit': 'month', 'type': 'temporal'}
    True

    >>> parse_shorthand('year(col):O') == {'field': 'col', 'timeUnit': 'year', 'type': 'ordinal'}
    True

    >>> parse_shorthand('foo', data) == {'field': 'foo', 'type': 'nominal'}
    True

    >>> parse_shorthand('bar', data) == {'field': 'bar', 'type': 'quantitative'}
    True

    >>> parse_shorthand('bar:O', data) == {'field': 'bar', 'type': 'ordinal'}
    True

    >>> parse_shorthand('sum(bar)', data) == {'aggregate': 'sum', 'field': 'bar', 'type': 'quantitative'}
    True

    >>> parse_shorthand('count()', data) == {'aggregate': 'count', 'type': 'quantitative'}
    True
    """
    from altair.utils._importers import pyarrow_available
    if not shorthand:
        return {}
    valid_typecodes = list(TYPECODE_MAP) + list(INV_TYPECODE_MAP)
    units = {'field': '(?P<field>.*)', 'type': '(?P<type>{})'.format('|'.join(valid_typecodes)), 'agg_count': '(?P<aggregate>count)', 'op_count': '(?P<op>count)', 'aggregate': '(?P<aggregate>{})'.format('|'.join(AGGREGATES)), 'window_op': '(?P<op>{})'.format('|'.join(AGGREGATES + WINDOW_AGGREGATES)), 'timeUnit': '(?P<timeUnit>{})'.format('|'.join(TIMEUNITS))}
    patterns = []
    if parse_aggregates:
        patterns.extend(['{agg_count}\\(\\)'])
        patterns.extend(['{aggregate}\\({field}\\)'])
    if parse_window_ops:
        patterns.extend(['{op_count}\\(\\)'])
        patterns.extend(['{window_op}\\({field}\\)'])
    if parse_timeunits:
        patterns.extend(['{timeUnit}\\({field}\\)'])
    patterns.extend(['{field}'])
    if parse_types:
        patterns = list(itertools.chain(*((p + ':{type}', p) for p in patterns)))
    regexps = (re.compile('\\A' + p.format(**units) + '\\Z', re.DOTALL) for p in patterns)
    if isinstance(shorthand, dict):
        attrs = shorthand
    else:
        attrs = next((exp.match(shorthand).groupdict() for exp in regexps if exp.match(shorthand) is not None))
    if 'type' in attrs:
        attrs['type'] = INV_TYPECODE_MAP.get(attrs['type'], attrs['type'])
    if attrs == {'aggregate': 'count'}:
        attrs['type'] = 'quantitative'
    if 'timeUnit' in attrs and 'type' not in attrs:
        attrs['type'] = 'temporal'
    if 'type' not in attrs:
        if pyarrow_available() and data is not None and isinstance(data, DataFrameLike):
            dfi = data.__dataframe__()
            if 'field' in attrs:
                unescaped_field = attrs['field'].replace('\\', '')
                if unescaped_field in dfi.column_names():
                    column = dfi.get_column_by_name(unescaped_field)
                    try:
                        attrs['type'] = infer_vegalite_type_for_dfi_column(column)
                    except (NotImplementedError, AttributeError, ValueError):
                        if isinstance(data, pd.DataFrame):
                            attrs['type'] = infer_vegalite_type(data[unescaped_field])
                        else:
                            raise
                    if isinstance(attrs['type'], tuple):
                        attrs['sort'] = attrs['type'][1]
                        attrs['type'] = attrs['type'][0]
        elif isinstance(data, pd.DataFrame):
            if 'field' in attrs and attrs['field'].replace('\\', '') in data.columns:
                attrs['type'] = infer_vegalite_type(data[attrs['field'].replace('\\', '')])
                if isinstance(attrs['type'], tuple):
                    attrs['sort'] = attrs['type'][1]
                    attrs['type'] = attrs['type'][0]
    if 'field' in attrs and ':' in attrs['field'] and (attrs['field'][attrs['field'].rfind(':') - 1] != '\\'):
        raise ValueError('"{}" '.format(attrs['field'].split(':')[-1]) + 'is not one of the valid encoding data types: {}.'.format(', '.join(TYPECODE_MAP.values())) + '\nFor more details, see https://altair-viz.github.io/user_guide/encodings/index.html#encoding-data-types. ' + 'If you are trying to use a column name that contains a colon, ' + 'prefix it with a backslash; for example "column\\:name" instead of "column:name".')
    return attrs