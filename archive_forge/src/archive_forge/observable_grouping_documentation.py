from typing import Iterable, Dict, List, TYPE_CHECKING, cast, Callable
from cirq import ops, value
from cirq.work.observable_settings import InitObsSetting, _max_weight_state, _max_weight_observable
Greedily group settings which can be simultaneously measured.

    We construct a dictionary keyed by `max_setting` (see docstrings
    for `_max_weight_state` and `_max_weight_observable`) where the value
    is a list of settings compatible with `max_setting`. For each new setting,
    we try to find an existing group to add it and update `max_setting` for
    that group if necessary. Otherwise, we make a new group.

    In practice, this greedy algorithm performs comparably to something
    more complicated by solving the clique cover problem on a graph
    of simultaneously-measurable settings.

    Args:
        settings: The settings to group.

    Returns:
        A dictionary keyed by `max_setting` which need not exist in the
        input list of settings. Each dictionary value is a list of
        settings compatible with `max_setting`.
    