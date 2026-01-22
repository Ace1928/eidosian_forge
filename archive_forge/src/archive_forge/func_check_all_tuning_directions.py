import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def check_all_tuning_directions(self, func: Callable[['triton.Config'], float], best_config, best_timing):
    """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
    candidate_values_list = []
    effective_fields = []
    for field in self.tunable_fields:
        old_value = get_field(best_config, field)
        if old_value is None:
            continue
        candidate_values = self.get_neighbour_values(field, old_value, radius=inductor_config.coordinate_descent_search_radius, include_self=True)
        candidate_values_list.append(candidate_values)
        effective_fields.append(field)
    choices = itertools.product(*candidate_values_list)
    improved = False
    for choice in choices:
        assert len(choice) == len(effective_fields)
        candidate_config = copy.deepcopy(best_config)
        for new_val, field in zip(choice, effective_fields):
            set_field(candidate_config, field, new_val)
        cmp_res, candidate_timing = self.compare_config(func, candidate_config, best_config, best_timing)
        if cmp_res:
            improved = True
            best_config = candidate_config
            best_timing = candidate_timing
    return (improved, best_config, best_timing)