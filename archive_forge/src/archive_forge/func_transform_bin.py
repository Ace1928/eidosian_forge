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
def transform_bin(self, as_: Union[str, core.FieldName, List[Union[str, core.FieldName]], UndefinedType]=Undefined, field: Union[str, core.FieldName, UndefinedType]=Undefined, bin: Union[Literal[True], core.BinParams]=True, **kwargs) -> Self:
    """
        Add a :class:`BinTransform` to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            The output fields at which to write the start and end bin values.
        bin : anyOf(boolean, :class:`BinParams`)
            An object indicating bin properties, or simply ``true`` for using default bin
            parameters.
        field : string
            The data field to bin.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_bin("x_binned", "x")
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: True,
          field: 'x'
        })

        >>> chart = alt.Chart().transform_bin("x_binned", "x",
        ...                                   bin=alt.Bin(maxbins=10))
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: BinParams({
            maxbins: 10
          }),
          field: 'x'
        })

        See Also
        --------
        alt.BinTransform : underlying transform object

        """
    if as_ is not Undefined:
        if 'as' in kwargs:
            raise ValueError("transform_bin: both 'as_' and 'as' passed as arguments.")
        kwargs['as'] = as_
    kwargs['bin'] = bin
    kwargs['field'] = field
    return self._add_transform(core.BinTransform(**kwargs))