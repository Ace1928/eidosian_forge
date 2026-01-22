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
def transform_quantile(self, quantile: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, probs: Union[List[float], UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined) -> Self:
    """Add a :class:`QuantileTransform` to the chart

        Parameters
        ----------
        quantile : str
            The data field for which to perform quantile estimation.
        as : [str, str]
            The output field names for the probability and quantile values.
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        probs : List(float)
            An array of probabilities in the range (0, 1) for which to compute quantile values.
            If not specified, the *step* parameter will be used.
        step : float
            A probability step size (default 0.01) for sampling quantile values. All values from
            one-half the step size up to 1 (exclusive) will be sampled. This parameter is only
            used if the *probs* parameter is not provided. **Default value:** ``["prob", "value"]``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.QuantileTransform : underlying transform object
        """
    return self._add_transform(core.QuantileTransform(quantile=quantile, groupby=groupby, probs=probs, step=step, **{'as': as_}))