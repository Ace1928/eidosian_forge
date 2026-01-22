from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def validate_cachify(self, func: str, **kwargs) -> Dict[str, Any]:
    """
        Validates the cachify function
        """
    if not self.cachify_enabled:
        return None
    from .utils import create_cachify_build_name_func
    base_name = self.cachify_create_base_name(func, **kwargs)
    if 'name' not in kwargs:
        kwargs['name'] = create_cachify_build_name_func(base_name=base_name, **self.cachify_get_name_builder_kwargs(func, **kwargs))
    if 'ttl' not in kwargs:
        kwargs['ttl'] = self.cachify_ttl
    if 'serializer' not in kwargs and 'encoder' not in kwargs and ('decoder' not in kwargs):
        kwargs['serializer'] = self.serialization
        kwargs['serializer_kwargs'] = {'compression': self.serialization_compression, 'compression_level': self.serialization_compression_level, 'raise_errors': True, **self.cachify_get_extra_serialization_kwargs(func, **kwargs)}
    if 'verbosity' not in kwargs and self.settings.is_local_env:
        kwargs['verbosity'] = 2
    kwargs['disabled'] = self.cachify_validator_is_not_cachable
    kwargs['overwrite_if'] = self.cachify_validator_is_overwrite
    kwargs['disabled_if'] = self.cachify_validator_is_disabled
    if (exclude_keys := self.cachify_get_exclude_keys(func, **kwargs)):
        kwargs['exclude_keys'] = exclude_keys
    kwargs['exclude_null'] = True
    return kwargs