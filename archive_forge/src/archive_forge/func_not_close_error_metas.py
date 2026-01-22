import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def not_close_error_metas(actual: Any, expected: Any, *, pair_types: Sequence[Type[Pair]]=(ObjectPair,), sequence_types: Tuple[Type, ...]=(collections.abc.Sequence,), mapping_types: Tuple[Type, ...]=(collections.abc.Mapping,), **options: Any) -> List[ErrorMeta]:
    """Asserts that inputs are equal.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s or
    :class:`~collections.abc.Mapping`'s. In this case the comparison happens elementwise by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of :class:`Pair` types that will be tried to construct with the
            inputs. First successful pair will be used. Defaults to only using :class:`ObjectPair`.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        **options (Any): Options passed to each pair during construction.
    """
    __tracebackhide__ = True
    try:
        pairs = originate_pairs(actual, expected, pair_types=pair_types, sequence_types=sequence_types, mapping_types=mapping_types, **options)
    except ErrorMeta as error_meta:
        raise error_meta.to_error() from None
    error_metas: List[ErrorMeta] = []
    for pair in pairs:
        try:
            pair.compare()
        except ErrorMeta as error_meta:
            error_metas.append(error_meta)
        except Exception as error:
            raise RuntimeError(f'Comparing\n\n{pair}\n\nresulted in the unexpected exception above. If you are a user and see this message during normal operation please file an issue at https://github.com/pytorch/pytorch/issues. If you are a developer and working on the comparison functions, please except the previous error and raise an expressive `ErrorMeta` instead.') from error
    error_metas = [error_metas]
    return error_metas.pop()