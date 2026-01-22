from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import islice, tee, zip_longest
from django.utils.functional import Promise
def normalize_choices(value, *, depth=0):
    """Normalize choices values consistently for fields and widgets."""
    from django.db.models.enums import ChoicesType
    match value:
        case BaseChoiceIterator() | Promise() | bytes() | str():
            return value
        case ChoicesType():
            return value.choices
        case Mapping() if depth < 2:
            value = value.items()
        case Iterator() if depth < 2:
            pass
        case Iterable() if depth < 2 and (not any((isinstance(x, (Promise, bytes, str)) for x in value))):
            pass
        case Callable() if depth == 0:
            return CallableChoiceIterator(value)
        case Callable() if depth < 2:
            value = value()
        case _:
            return value
    try:
        return [(k, normalize_choices(v, depth=depth + 1)) for k, v in value]
    except (TypeError, ValueError):
        return value