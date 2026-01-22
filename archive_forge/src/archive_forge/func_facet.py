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
def facet(self, facet: Union[str, channels.Facet, UndefinedType]=Undefined, row: Union[str, core.FacetFieldDef, channels.Row, UndefinedType]=Undefined, column: Union[str, core.FacetFieldDef, channels.Column, UndefinedType]=Undefined, data: Union[ChartDataType, UndefinedType]=Undefined, columns: Union[int, UndefinedType]=Undefined, **kwargs) -> 'FacetChart':
    """Create a facet chart from the current chart.

        Faceted charts require data to be specified at the top level; if data
        is not specified, the data from the current chart will be used at the
        top level.

        Parameters
        ----------
        facet : string, Facet (optional)
            The data column to use as an encoding for a wrapped facet.
            If specified, then neither row nor column may be specified.
        column : string, Column, FacetFieldDef (optional)
            The data column to use as an encoding for a column facet.
            May be combined with row argument, but not with facet argument.
        row : string or Row, FacetFieldDef (optional)
            The data column to use as an encoding for a row facet.
            May be combined with column argument, but not with facet argument.
        data : string or dataframe (optional)
            The dataset to use for faceting. If not supplied, then data must
            be specified in the top-level chart that calls this method.
        columns : integer
            the maximum number of columns for a wrapped facet.

        Returns
        -------
        self :
            for chaining
        """
    facet_specified = facet is not Undefined
    rowcol_specified = row is not Undefined or column is not Undefined
    if facet_specified and rowcol_specified:
        raise ValueError('facet argument cannot be combined with row/column argument.')
    if data is Undefined:
        if self.data is Undefined:
            raise ValueError('Facet charts require data to be specified at the top level. If you are trying to facet layered or concatenated charts, ensure that the same data variable is passed to each chart or specify the data inside the facet method instead.')
        self = self.copy(deep=False)
        data, self.data = (self.data, Undefined)
    if facet_specified:
        if isinstance(facet, str):
            facet = channels.Facet(facet)
    else:
        facet = FacetMapping(row=row, column=column)
    return FacetChart(spec=self, facet=facet, data=data, columns=columns, **kwargs)