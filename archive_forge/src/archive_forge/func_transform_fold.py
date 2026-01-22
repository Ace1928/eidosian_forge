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
def transform_fold(self, fold: List[Union[str, core.FieldName]], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined) -> Self:
    """Add a :class:`FoldTransform` to the spec.

        Parameters
        ----------
        fold : List(string)
            An array of data fields indicating the properties to fold.
        as : [string, string]
            The output field names for the key and value properties produced by the fold
            transform. Default: ``["key", "value"]``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_pivot : pivot transform - opposite of fold.
        alt.FoldTransform : underlying transform object
        """
    return self._add_transform(core.FoldTransform(fold=fold, **{'as': as_}))