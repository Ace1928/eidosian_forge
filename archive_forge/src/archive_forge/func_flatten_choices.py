from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import islice, tee, zip_longest
from django.utils.functional import Promise
def flatten_choices(choices):
    """Flatten choices by removing nested values."""
    for value_or_group, label_or_nested in choices or ():
        if isinstance(label_or_nested, (list, tuple)):
            yield from label_or_nested
        else:
            yield (value_or_group, label_or_nested)