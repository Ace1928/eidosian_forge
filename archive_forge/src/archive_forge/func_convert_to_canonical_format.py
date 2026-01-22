import functools
import logging
from collections import abc
from typing import Union, Mapping, Any, Callable
from ray.rllib.core.models.specs.specs_base import Spec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.nested_dict import NestedDict
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def convert_to_canonical_format(spec: SpecType) -> Union[Spec, SpecDict]:
    """Converts a spec type input to the canonical format.

    The canonical format is either

    1. A nested SpecDict when the input spec is dict-like tree of specs and types or
    nested list of nested_keys.
    2. A single SpecType object, if the spec is a single constraint.

    The input can be any of the following:
        - a list of nested_keys. nested_keys are either strings or tuples of strings
        specifying the path to a leaf in the tree.
        - a tree of constraints. The tree structure can be specified as any nested
        hash-map structure (e.g. dict, SpecDict, NestedDict, etc.) The leaves of the
        tree can be either a Spec object, a type, or None. If the leaf is a type, it is
        converted to a TypeSpec. If the leaf is None, only the existance of the key is
        checked and the value will be None in the canonical format.
        - a single constraint. The constraint can be a Spec object, a type, or None.


        Examples of canoncial format #1:

        >>> spec = {'foo': int, 'bar': {'baz': None}}
        >>> convert_to_canonical_format(spec)
        SpecDict({'foo': TypeSpec(<class 'int'>), 'bar': SpecDict({'baz': None})})

        >>> spec = ['foo', ('bar', 'baz')]
        >>> convert_to_canonical_format(spec)
        SpecDict({'foo': None, 'bar': SpecDict({'baz': None})})

        >>> from ray.rllib.core.models.specs.specs_base import TensorSpec
        >>> spec = {'bar': {'baz': TensorSpec('b,h', framework='torch')}}
        >>> convert_to_canonical_format(spec)
        SpecDict({'bar': SpecDict({'baz': TensorSpec(shape=('b', 'h'), dtype=None)})})

        Example of canoncial format #2:

        >>> from ray.rllib.core.models.specs.specs_base import TensorSpec

        >>> spec = int
        >>> convert_to_canonical_format(spec)
        TypeSpec(<class 'int'>)

        >>> spec = None
        >>> convert_to_canonical_format(spec) # Returns None

        >>> spec = TensorSpec('b,h', framework='torch')
        >>> convert_to_canonical_format(spec)
        TensorSpec(shape=('b', 'h'), dtype=None)

    Args:
        spec: The spec to convert to canonical format.

    Returns:
        The canonical format of the spec.
    """
    if isinstance(spec, list):
        spec = [(k,) if isinstance(k, str) else k for k in spec]
        return SpecDict({k: None for k in spec})
    if isinstance(spec, abc.Mapping):
        spec = SpecDict(spec)
        for key in spec:
            if isinstance(spec[key], (type, tuple)):
                spec[key] = TypeSpec(spec[key])
            elif isinstance(spec[key], list):
                spec[key] = convert_to_canonical_format(spec[key])
        return spec
    if isinstance(spec, type):
        return TypeSpec(spec)
    return spec