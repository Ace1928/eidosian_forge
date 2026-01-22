from __future__ import annotations
import collections, functools, importlib
import typing as T
from .base import ExternalDependency, DependencyException, DependencyMethods, NotFoundDependency
from ..mesonlib import listify, MachineChoice, PerMachine
from .. import mlog
def get_dep_identifier(name: str, kwargs: T.Dict[str, T.Any]) -> 'TV_DepID':
    identifier: 'TV_DepID' = (('name', name),)
    from ..interpreter import permitted_dependency_kwargs
    assert len(permitted_dependency_kwargs) == 19, 'Extra kwargs have been added to dependency(), please review if it makes sense to handle it here'
    for key, value in kwargs.items():
        if key in {'version', 'native', 'required', 'fallback', 'allow_fallback', 'default_options', 'not_found_message', 'include_type'}:
            continue
        if isinstance(value, list):
            for i in value:
                assert isinstance(i, str)
            value = tuple(frozenset(listify(value)))
        else:
            assert isinstance(value, (str, bool, int))
        identifier = (*identifier, (key, value))
    return identifier