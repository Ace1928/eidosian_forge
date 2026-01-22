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
def transform_flatten(self, flatten: List[Union[str, core.FieldName]], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined) -> Self:
    """Add a :class:`FlattenTransform` to the schema.

        Parameters
        ----------
        flatten : List(string)
            An array of one or more data fields containing arrays to flatten.
            If multiple fields are specified, their array values should have a parallel
            structure, ideally with the same length.
            If the lengths of parallel arrays do not match,
            the longest array will be used with ``null`` values added for missing entries.
        as : List(string)
            The output field names for extracted array values.
            **Default value:** The field name of the corresponding array field

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.FlattenTransform : underlying transform object
        """
    return self._add_transform(core.FlattenTransform(flatten=flatten, **{'as': as_}))