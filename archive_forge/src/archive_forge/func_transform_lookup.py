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
def transform_lookup(self, lookup: Union[str, UndefinedType]=Undefined, from_: Union[core.LookupData, core.LookupSelection, UndefinedType]=Undefined, as_: Union[Union[str, core.FieldName], List[Union[str, core.FieldName]], UndefinedType]=Undefined, default: Union[str, UndefinedType]=Undefined, **kwargs) -> Self:
    """Add a :class:`DataLookupTransform` or :class:`SelectionLookupTransform` to the chart

        Parameters
        ----------
        lookup : string
            Key in primary data source.
        from_ : anyOf(:class:`LookupData`, :class:`LookupSelection`)
            Secondary data reference.
        as_ : anyOf(string, List(string))
            The output fields on which to store the looked up data values.

            For data lookups, this property may be left blank if ``from_.fields``
            has been specified (those field names will be used); if ``from_.fields``
            has not been specified, ``as_`` must be a string.

            For selection lookups, this property is optional: if unspecified,
            looked up values will be stored under a property named for the selection;
            and if specified, it must correspond to ``from_.fields``.
        default : string
            The default value to use if lookup fails. **Default value:** ``null``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.DataLookupTransform : underlying transform object
        alt.SelectionLookupTransform : underlying transform object
        """
    if as_ is not Undefined:
        if 'as' in kwargs:
            raise ValueError("transform_lookup: both 'as_' and 'as' passed as arguments.")
        kwargs['as'] = as_
    if from_ is not Undefined:
        if 'from' in kwargs:
            raise ValueError("transform_lookup: both 'from_' and 'from' passed as arguments.")
        kwargs['from'] = from_
    kwargs['lookup'] = lookup
    kwargs['default'] = default
    return self._add_transform(core.LookupTransform(**kwargs))