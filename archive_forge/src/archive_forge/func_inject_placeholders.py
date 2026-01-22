from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def inject_placeholders(config: Any, resolvers: defaultdict, id_prefix: Tuple=(), path_prefix: Tuple=()) -> Dict:
    """Replaces reference objects contained by a config dict with placeholders.

    Given a config dict, this function replaces all reference objects contained
    by this dict with placeholder strings. It recursively expands nested dicts
    and lists, and properly handles Tune native search objects such as Categorical
    and Function.
    This makes sure the config dict only contains primitive typed values, which
    can then be handled by different search algorithms.

    A few details about id_prefix and path_prefix. Consider the following config,
    where "param1" is a simple grid search of 3 tuples.

    config = {
        "param1": tune.grid_search([
            (Cat, None, None),
            (None, Dog, None),
            (None, None, Fish),
        ]),
    }

    We will replace the 3 objects contained with placeholders. And after trial
    expansion, the config may look like this:

    config = {
        "param1": (None, (placeholder, hash), None)
    }

    Now you need 2 pieces of information to resolve the placeholder. One is the
    path of ("param1", 1), which tells you that the first element of the tuple
    under "param1" key is a placeholder that needs to be resolved.
    The other is the mapping from the placeholder to the actual object. In this
    case hash -> Dog.

    id and path prefixes serve exactly this purpose here. The difference between
    these two is that id_prefix is the location of the value in the pre-injected
    config tree. So if a value is the second option in a grid_search, it gets an
    id part of 1. Injected placeholders all get unique id prefixes. path prefix
    identifies a placeholder in the expanded config tree. So for example, all
    options of a single grid_search will get the same path prefix. This is how
    we know which location has a placeholder to be resolved in the post-expansion
    tree.

    Args:
        config: The config dict to replace references in.
        resolvers: A dict from path to replaced objects.
        id_prefix: The prefix to prepend to id every single placeholders.
        path_prefix: The prefix to prepend to every path identifying
            potential locations of placeholders in an expanded tree.

    Returns:
        The config with all references replaced.
    """
    if isinstance(config, dict) and 'grid_search' in config and (len(config) == 1):
        config['grid_search'] = [inject_placeholders(choice, resolvers, id_prefix + (i,), path_prefix) for i, choice in enumerate(config['grid_search'])]
        return config
    elif isinstance(config, dict):
        return {k: inject_placeholders(v, resolvers, id_prefix + (k,), path_prefix + (k,)) for k, v in config.items()}
    elif isinstance(config, list):
        return [inject_placeholders(elem, resolvers, id_prefix + (i,), path_prefix + (i,)) for i, elem in enumerate(config)]
    elif isinstance(config, tuple):
        return tuple((inject_placeholders(elem, resolvers, id_prefix + (i,), path_prefix + (i,)) for i, elem in enumerate(config)))
    elif _is_primitive(config):
        return config
    elif isinstance(config, Categorical):
        config.categories = [inject_placeholders(choice, resolvers, id_prefix + (i,), path_prefix) for i, choice in enumerate(config.categories)]
        return config
    elif isinstance(config, Function):
        id_hash = _id_hash(id_prefix)
        v = _FunctionResolver(id_hash, config)
        resolvers[path_prefix].append(v)
        return v.get_placeholder()
    elif not isinstance(config, Domain):
        id_hash = _id_hash(id_prefix)
        v = _RefResolver(id_hash, config)
        resolvers[path_prefix].append(v)
        return v.get_placeholder()
    else:
        return config