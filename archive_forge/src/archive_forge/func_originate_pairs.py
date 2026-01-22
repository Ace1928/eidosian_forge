import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def originate_pairs(actual: Any, expected: Any, *, pair_types: Sequence[Type[Pair]], sequence_types: Tuple[Type, ...]=(collections.abc.Sequence,), mapping_types: Tuple[Type, ...]=(collections.abc.Mapping,), id: Tuple[Any, ...]=(), **options: Any) -> List[Pair]:
    """Originates pairs from the individual inputs.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s or
    :class:`~collections.abc.Mapping`'s. In this case the pairs are originated by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of pair types that will be tried to construct with the inputs.
            First successful pair will be used.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        id (Tuple[Any, ...]): Optional id of a pair that will be included in an error message.
        **options (Any): Options passed to each pair during construction.

    Raises:
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Sequence`'s, but their
            length does not match.
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Mapping`'s, but their set of
            keys do not match.
        ErrorMeta: With :class`TypeError`, if no pair is able to handle the inputs.
        ErrorMeta: With any expected exception that happens during the construction of a pair.

    Returns:
        (List[Pair]): Originated pairs.
    """
    if isinstance(actual, sequence_types) and (not isinstance(actual, str)) and isinstance(expected, sequence_types) and (not isinstance(expected, str)):
        actual_len = len(actual)
        expected_len = len(expected)
        if actual_len != expected_len:
            raise ErrorMeta(AssertionError, f'The length of the sequences mismatch: {actual_len} != {expected_len}', id=id)
        pairs = []
        for idx in range(actual_len):
            pairs.extend(originate_pairs(actual[idx], expected[idx], pair_types=pair_types, sequence_types=sequence_types, mapping_types=mapping_types, id=(*id, idx), **options))
        return pairs
    elif isinstance(actual, mapping_types) and isinstance(expected, mapping_types):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            missing_keys = expected_keys - actual_keys
            additional_keys = actual_keys - expected_keys
            raise ErrorMeta(AssertionError, f'The keys of the mappings do not match:\nMissing keys in the actual mapping: {sorted(missing_keys)}\nAdditional keys in the actual mapping: {sorted(additional_keys)}', id=id)
        keys: Collection = actual_keys
        with contextlib.suppress(Exception):
            keys = sorted(keys)
        pairs = []
        for key in keys:
            pairs.extend(originate_pairs(actual[key], expected[key], pair_types=pair_types, sequence_types=sequence_types, mapping_types=mapping_types, id=(*id, key), **options))
        return pairs
    else:
        for pair_type in pair_types:
            try:
                return [pair_type(actual, expected, id=id, **options)]
            except UnsupportedInputs:
                continue
            except ErrorMeta:
                raise
            except Exception as error:
                raise RuntimeError(f'Originating a {pair_type.__name__}() at item {''.join((str([item]) for item in id))} with\n\n{type(actual).__name__}(): {actual}\n\nand\n\n{type(expected).__name__}(): {expected}\n\nresulted in the unexpected exception above. If you are a user and see this message during normal operation please file an issue at https://github.com/pytorch/pytorch/issues. If you are a developer and working on the comparison functions, please except the previous error and raise an expressive `ErrorMeta` instead.') from error
        else:
            raise ErrorMeta(TypeError, f'No comparison pair was able to handle inputs of type {type(actual)} and {type(expected)}.', id=id)