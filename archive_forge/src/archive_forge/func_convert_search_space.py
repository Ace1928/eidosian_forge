import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
@staticmethod
def convert_search_space(spec: Dict) -> Dict[str, Any]:
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)
    if not domain_vars and (not grid_vars):
        return {}
    if grid_vars:
        raise ValueError('Grid search parameters cannot be automatically converted to an Optuna search space.')
    spec = flatten_dict(spec, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

    def resolve_value(domain: Domain) -> ot.distributions.BaseDistribution:
        quantize = None
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            quantize = sampler.q
            sampler = sampler.sampler
            if isinstance(sampler, LogUniform):
                logger.warning('Optuna does not handle quantization in loguniform sampling. The parameter will be passed but it will probably be ignored.')
        if isinstance(domain, Float):
            if isinstance(sampler, LogUniform):
                if quantize:
                    logger.warning('Optuna does not support both quantization and sampling from LogUniform. Dropped quantization.')
                return ot.distributions.FloatDistribution(domain.lower, domain.upper, log=True)
            elif isinstance(sampler, Uniform):
                if quantize:
                    return ot.distributions.FloatDistribution(domain.lower, domain.upper, step=quantize)
                return ot.distributions.FloatDistribution(domain.lower, domain.upper)
        elif isinstance(domain, Integer):
            if isinstance(sampler, LogUniform):
                return ot.distributions.IntDistribution(domain.lower, domain.upper - 1, step=quantize or 1, log=True)
            elif isinstance(sampler, Uniform):
                return ot.distributions.IntDistribution(domain.lower, domain.upper - int(bool(not quantize)), step=quantize or 1)
        elif isinstance(domain, Categorical):
            if isinstance(sampler, Uniform):
                return ot.distributions.CategoricalDistribution(domain.categories)
        raise ValueError('Optuna search does not support parameters of type `{}` with samplers of type `{}`'.format(type(domain).__name__, type(domain.sampler).__name__))
    values = {'/'.join(path): resolve_value(domain) for path, domain in domain_vars}
    return values