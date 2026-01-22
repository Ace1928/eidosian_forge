from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def resolve_placeholders(config: Any, replaced: defaultdict):
    """Replaces placeholders contained by a config dict with the original values.

    Args:
        config: The config to replace placeholders in.
        replaced: A dict from path to replaced objects.
    """

    def __resolve(resolver_type, args):
        for path, resolvers in replaced.items():
            assert resolvers
            if not isinstance(resolvers[0], resolver_type):
                continue
            prefix, ph = _get_placeholder(config, (), path)
            if not ph:
                continue
            for resolver in resolvers:
                if resolver.hash != ph[1]:
                    continue
                assign_value(config, prefix, resolver.resolve(*args))
    __resolve(_RefResolver, args=())
    __resolve(_FunctionResolver, args=(config,))