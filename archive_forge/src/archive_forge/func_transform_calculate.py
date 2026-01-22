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
def transform_calculate(self, as_: Union[str, core.FieldName, UndefinedType]=Undefined, calculate: Union[str, core.Expr, _expr_core.Expression, UndefinedType]=Undefined, **kwargs: Union[str, core.Expr, _expr_core.Expression]) -> Self:
    """
        Add a :class:`CalculateTransform` to the schema.

        Parameters
        ----------
        as_ : string
            The field for storing the computed formula value.
        calculate : string or alt.expr.Expression
            An `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
            string. Use the variable ``datum`` to refer to the current data object.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_calculate(y = 2 * expr.sin(datum.x))
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: (2 * sin(datum.x))
        })

        It's also possible to pass the ``CalculateTransform`` arguments directly:

        >>> kwds = {'as_': 'y', 'calculate': '2 * sin(datum.x)'}
        >>> chart = alt.Chart().transform_calculate(**kwds)
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: '2 * sin(datum.x)'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.CalculateTransform : underlying transform object
        """
    if as_ is Undefined:
        as_ = kwargs.pop('as', Undefined)
    elif 'as' in kwargs:
        raise ValueError("transform_calculate: both 'as_' and 'as' passed as arguments.")
    if as_ is not Undefined or calculate is not Undefined:
        dct = {'as': as_, 'calculate': calculate}
        self = self._add_transform(core.CalculateTransform(**dct))
    for as_, calculate in kwargs.items():
        dct = {'as': as_, 'calculate': calculate}
        self = self._add_transform(core.CalculateTransform(**dct))
    return self