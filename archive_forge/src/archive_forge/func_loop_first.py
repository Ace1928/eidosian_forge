from typing import Iterable, Tuple, TypeVar
def loop_first(values: Iterable[T]) -> Iterable[Tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for first value."""
    iter_values = iter(values)
    try:
        value = next(iter_values)
    except StopIteration:
        return
    yield (True, value)
    for value in iter_values:
        yield (False, value)