import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def transform_joinaggregate(self, joinaggregate: Union[List[core.JoinAggregateFieldDef], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, **kwargs: str) -> Self:
    """
        Add a :class:`JoinAggregateTransform` to the schema.

        Parameters
        ----------
        joinaggregate : List(:class:`JoinAggregateFieldDef`)
            The definition of the fields in the join aggregate, and what calculations to use.
        groupby : List(string)
            The data fields for partitioning the data objects into separate groups. If
            unspecified, all data points will be in a single group.
        **kwargs
            joinaggregates can also be passed by keyword argument; see Examples.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_joinaggregate(x='sum(y)')
        >>> chart.transform[0]
        JoinAggregateTransform({
          joinaggregate: [JoinAggregateFieldDef({
            as: 'x',
            field: 'y',
            op: 'sum'
          })]
        })

        See Also
        --------
        alt.JoinAggregateTransform : underlying transform object
        """
    if joinaggregate is Undefined:
        joinaggregate = []
    for key, val in kwargs.items():
        parsed = utils.parse_shorthand(val)
        dct = {'as': key, 'field': parsed.get('field', Undefined), 'op': parsed.get('aggregate', Undefined)}
        assert not isinstance(joinaggregate, UndefinedType)
        joinaggregate.append(core.JoinAggregateFieldDef(**dct))
    return self._add_transform(core.JoinAggregateTransform(joinaggregate=joinaggregate, groupby=groupby))