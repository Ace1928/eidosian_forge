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
def transform_impute(self, impute: Union[str, core.FieldName], key: Union[str, core.FieldName], frame: Union[List[Optional[int]], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, keyvals: Union[List[Any], core.ImputeSequence, UndefinedType]=Undefined, method: Union[Literal['value', 'mean', 'median', 'max', 'min'], core.ImputeMethod, UndefinedType]=Undefined, value=Undefined) -> Self:
    """
        Add an :class:`ImputeTransform` to the schema.

        Parameters
        ----------
        impute : string
            The data field for which the missing values should be imputed.
        key : string
            A key field that uniquely identifies data objects within a group.
            Missing key values (those occurring in the data but not in the current group) will
            be imputed.
        frame : List(anyOf(None, int))
            A frame specification as a two-element array used to control the window over which
            the specified method is applied. The array entries should either be a number
            indicating the offset from the current data object, or null to indicate unbounded
            rows preceding or following the current data object.  For example, the value ``[-5,
            5]`` indicates that the window should include five objects preceding and five
            objects following the current object.
            **Default value:** :  ``[null, null]`` indicating that the window includes all
            objects.
        groupby : List(string)
            An optional array of fields by which to group the values.
            Imputation will then be performed on a per-group basis.
        keyvals : anyOf(List(Mapping(required=[])), :class:`ImputeSequence`)
            Defines the key values that should be considered for imputation.
            An array of key values or an object defining a `number sequence
            <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.
            If provided, this will be used in addition to the key values observed within the
            input data.  If not provided, the values will be derived from all unique values of
            the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
            the y-field is imputed, or vice versa.
            If there is no impute grouping, this property *must* be specified.
        method : :class:`ImputeMethod`
            The imputation method to use for the field value of imputed data objects.
            One of ``value``, ``mean``, ``median``, ``max`` or ``min``.
            **Default value:**  ``"value"``
        value : Mapping(required=[])
            The field value to use when the imputation ``method`` is ``"value"``.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.ImputeTransform : underlying transform object
        """
    return self._add_transform(core.ImputeTransform(impute=impute, key=key, frame=frame, groupby=groupby, keyvals=keyvals, method=method, value=value))