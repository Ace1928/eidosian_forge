import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, TypeVar, Union
from ray.util.annotations import Deprecated
@Deprecated
def unflatten_list_dict(dt: Dict[str, T], delimiter: str='/') -> Dict[str, T]:
    """Unflatten nested dict and list.

    This function now has some limitations:
    (1) The keys of dt must be str.
    (2) If unflattened dt (the result) contains list, the index order must be
        ascending when accessing dt. Otherwise, this function will throw
        AssertionError.
    (3) The unflattened dt (the result) shouldn't contain dict with number
        keys.

    Be careful to use this function. If you want to improve this function,
    please also improve the unit test. See #14487 for more details.

    Args:
        dt: Flattened dictionary that is originally nested by multiple
            list and dict.
        delimiter: Delimiter of keys.

    Example:
        >>> dt = {"aaa/0/bb": 12, "aaa/1/cc": 56, "aaa/1/dd": 92}
        >>> unflatten_list_dict(dt)
        {'aaa': [{'bb': 12}, {'cc': 56, 'dd': 92}]}
    """
    out_type = list if list(dt)[0].split(delimiter, 1)[0].isdigit() else type(dt)
    out = out_type()
    for key, val in dt.items():
        path = key.split(delimiter)
        item = out
        for i, k in enumerate(path[:-1]):
            next_type = list if path[i + 1].isdigit() else dict
            if isinstance(item, dict):
                item = item.setdefault(k, next_type())
            elif isinstance(item, list):
                if int(k) >= len(item):
                    item.append(next_type())
                    assert int(k) == len(item) - 1
                item = item[int(k)]
        if isinstance(item, dict):
            item[path[-1]] = val
        elif isinstance(item, list):
            item.append(val)
            assert int(path[-1]) == len(item) - 1
    return out