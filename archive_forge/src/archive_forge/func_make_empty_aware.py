from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def make_empty_aware(it: Union[Iterable[T], Iterator[T]]) -> 'EmptyAwareIterable[T]':
    """Make an iterable empty aware, or return itself if already empty aware

    :param it: underlying iterable

    :return: EmptyAwareIterable[T]
    """
    return it if isinstance(it, EmptyAwareIterable) else EmptyAwareIterable(it)