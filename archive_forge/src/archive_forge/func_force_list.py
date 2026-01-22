import contextlib
from functools import partial
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.filter_manager import FilterManager
from ray.rllib.utils.framework import (
from ray.rllib.utils.numpy import (
from ray.rllib.utils.pre_checks.env import check_env
from ray.rllib.utils.schedules import (
from ray.rllib.utils.test_utils import (
from ray.tune.utils import merge_dicts, deep_update
@DeveloperAPI
def force_list(elements=None, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single
    item, already a list, or a tuple.

    Args:
        elements (Optional[any]): The inputs as single item, list, or tuple to
            be converted into a list/tuple. If None, returns empty list/tuple.
        to_tuple: Whether to use tuple (instead of list).

    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on
            `to_tuple`'s value. If elements is None,
            returns an empty list/tuple.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) if type(elements) in [list, set, tuple] else ctor([elements])