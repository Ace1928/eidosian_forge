from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
@st.composite
def unique_subset_of(draw: st.DrawFn, objs: Union[Sequence[Hashable], Mapping[Hashable, Any]], *, min_size: int=0, max_size: Union[int, None]=None) -> Union[Sequence[Hashable], Mapping[Hashable, Any]]:
    """
    Return a strategy which generates a unique subset of the given objects.

    Each entry in the output subset will be unique (if input was a sequence) or have a unique key (if it was a mapping).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    objs: Union[Sequence[Hashable], Mapping[Hashable, Any]]
        Objects from which to sample to produce the subset.
    min_size: int, optional
        Minimum size of the returned subset. Default is 0.
    max_size: int, optional
        Maximum size of the returned subset. Default is the full length of the input.
        If set to 0 the result will be an empty mapping.

    Returns
    -------
    unique_subset_strategy
        Strategy generating subset of the input.

    Examples
    --------
    >>> unique_subset_of({"x": 2, "y": 3}).example()  # doctest: +SKIP
    {'y': 3}
    >>> unique_subset_of(["x", "y"]).example()  # doctest: +SKIP
    ['x']

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    if not isinstance(objs, Iterable):
        raise TypeError(f'Object to sample from must be an Iterable or a Mapping, but received type {type(objs)}')
    if len(objs) == 0:
        raise ValueError("Can't sample from a length-zero object.")
    keys = list(objs.keys()) if isinstance(objs, Mapping) else objs
    subset_keys = draw(st.lists(st.sampled_from(keys), unique=True, min_size=min_size, max_size=max_size))
    return {k: objs[k] for k in subset_keys} if isinstance(objs, Mapping) else subset_keys