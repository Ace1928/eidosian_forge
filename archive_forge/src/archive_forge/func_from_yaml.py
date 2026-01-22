from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def from_yaml(cls: Type[DataclassType], stream: Union[str, IO[str], bytes, IO[bytes]]) -> DataclassType:
    """Re-construct a dataclass instance from a yaml-compatible string, which should be
    generated from `tyro.extras.to_yaml()`.

    As a secondary feature aimed at enabling the use of :func:`tyro.cli` for general
    configuration use cases, we also introduce functions for human-readable dataclass
    serialization: :func:`tyro.extras.from_yaml` and :func:`tyro.extras.to_yaml` attempt
    to strike a balance between flexibility and robustness — in contrast to naively
    dumping or loading dataclass instances (via pickle, PyYAML, etc), explicit type
    references enable custom tags that are robust against code reorganization and
    refactor, while a PyYAML backend enables serialization of arbitrary Python objects.

    .. warning::
        Serialization functionality is stable but deprecated. It may be removed in a
        future version of :code:`tyro`.

    Args:
        cls: Type to reconstruct.
        stream: YAML to read from.

    Returns:
        Instantiated dataclass.
    """
    import yaml
    out = yaml.load(stream, Loader=_make_loader(cls))
    origin_cls = get_origin(cls)
    assert isinstance(out, origin_cls if origin_cls is not None else cls)
    return out