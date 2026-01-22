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
def transform_pivot(self, pivot: Union[str, core.FieldName], value: Union[str, core.FieldName], groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, limit: Union[int, UndefinedType]=Undefined, op: Union[str, core.AggregateOp, UndefinedType]=Undefined) -> Self:
    """Add a :class:`PivotTransform` to the chart.

        Parameters
        ----------
        pivot : str
            The data field to pivot on. The unique values of this field become new field names
            in the output stream.
        value : str
            The data field to populate pivoted fields. The aggregate values of this field become
            the values of the new pivoted fields.
        groupby : List(str)
            The optional data fields to group by. If not specified, a single group containing
            all data objects will be used.
        limit : int
            An optional parameter indicating the maximum number of pivoted fields to generate.
            The default ( ``0`` ) applies no limit. The pivoted ``pivot`` names are sorted in
            ascending order prior to enforcing the limit.
            **Default value:** ``0``
        op : string
            The aggregation operation to apply to grouped ``value`` field values.
            **Default value:** ``sum``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_fold : fold transform - opposite of pivot.
        alt.PivotTransform : underlying transform object
        """
    return self._add_transform(core.PivotTransform(pivot=pivot, value=value, groupby=groupby, limit=limit, op=op))