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
def transform_density(self, density: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, bandwidth: Union[float, UndefinedType]=Undefined, counts: Union[bool, UndefinedType]=Undefined, cumulative: Union[bool, UndefinedType]=Undefined, extent: Union[List[float], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, maxsteps: Union[int, UndefinedType]=Undefined, minsteps: Union[int, UndefinedType]=Undefined, steps: Union[int, UndefinedType]=Undefined) -> Self:
    """Add a :class:`DensityTransform` to the spec.

        Parameters
        ----------
        density : str
            The data field for which to perform density estimation.
        as_ : [str, str]
            The output fields for the sample value and corresponding density estimate.
            **Default value:** ``["value", "density"]``
        bandwidth : float
            The bandwidth (standard deviation) of the Gaussian kernel. If unspecified or set to
            zero, the bandwidth value is automatically estimated from the input data using
            Scott’s rule.
        counts : boolean
            A boolean flag indicating if the output values should be probability estimates
            (false) or smoothed counts (true).
            **Default value:** ``false``
        cumulative : boolean
            A boolean flag indicating whether to produce density estimates (false) or cumulative
            density estimates (true).
            **Default value:** ``false``
        extent : List([float, float])
            A [min, max] domain from which to sample the distribution. If unspecified, the
            extent will be determined by the observed minimum and maximum values of the density
            value field.
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        maxsteps : int
            The maximum number of samples to take along the extent domain for plotting the
            density. **Default value:** ``200``
        minsteps : int
            The minimum number of samples to take along the extent domain for plotting the
            density. **Default value:** ``25``
        steps : int
            The exact number of samples to take along the extent domain for plotting the
            density. If specified, overrides both minsteps and maxsteps to set an exact number
            of uniform samples. Potentially useful in conjunction with a fixed extent to ensure
            consistent sample points for stacked densities.
        """
    return self._add_transform(core.DensityTransform(density=density, bandwidth=bandwidth, counts=counts, cumulative=cumulative, extent=extent, groupby=groupby, maxsteps=maxsteps, minsteps=minsteps, steps=steps, **{'as': as_}))