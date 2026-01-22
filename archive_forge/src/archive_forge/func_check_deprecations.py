from __future__ import annotations
import ast
import base64
import builtins  # Explicitly use builtins.set as 'set' will be shadowed by a function
import json
import os
import pathlib
import site
import sys
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, overload
import yaml
from dask.typing import no_default
def check_deprecations(key: str, deprecations: Mapping[str, str | None]=deprecations) -> str:
    """Check if the provided value has been renamed or removed

    Parameters
    ----------
    key : str
        The configuration key to check
    deprecations : Dict[str, str]
        The mapping of aliases

    Examples
    --------
    >>> deprecations = {"old_key": "new_key", "invalid": None}
    >>> check_deprecations("old_key", deprecations=deprecations)  # doctest: +SKIP
    FutureWarning: Dask configuration key 'old_key' has been deprecated; please use "new_key" instead

    >>> check_deprecations("invalid", deprecations=deprecations)
    Traceback (most recent call last):
        ...
    ValueError: Dask configuration key 'invalid' has been removed

    >>> check_deprecations("another_key", deprecations=deprecations)
    'another_key'

    Returns
    -------
    new: str
        The proper key, whether the original (if no deprecation) or the aliased
        value

    See Also
    --------
    rename
    """
    old = key.replace('_', '-')
    if old in deprecations:
        new = deprecations[old]
        if new:
            warnings.warn(f'Dask configuration key {key!r} has been deprecated; please use {new!r} instead', FutureWarning)
            return new
        else:
            raise ValueError(f'Dask configuration key {key!r} has been removed')
    else:
        return key