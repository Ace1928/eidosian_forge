import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def compare_config(self, func, candidate_config, best_config, best_timing):
    """
        Check if candidate_config is better than best_config.

        Return a touple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
    log.debug('Try config %s', candidate_config)
    try:
        candidate_timing = self.call_func(func, candidate_config)
    except Exception as e:
        log.debug('Got exception %s', e)
        return (False, float('inf'))
    if self.has_improvement(best_timing, candidate_timing):
        log.debug('Tune from %s %f -> %s %f', best_config, best_timing, candidate_config, candidate_timing)
        return (True, candidate_timing)
    return (False, candidate_timing)