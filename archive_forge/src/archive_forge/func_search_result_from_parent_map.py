import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def search_result_from_parent_map(parent_map, missing_keys):
    """Transform a parent_map into SearchResult information."""
    if not parent_map:
        return ([], [], 0)
    start_set = set(parent_map)
    result_parents = set(itertools.chain.from_iterable(parent_map.values()))
    stop_keys = result_parents.difference(start_set)
    stop_keys.difference_update(missing_keys)
    key_count = len(parent_map)
    if revision.NULL_REVISION in result_parents and revision.NULL_REVISION in missing_keys:
        key_count += 1
    included_keys = start_set.intersection(result_parents)
    start_set.difference_update(included_keys)
    return (start_set, stop_keys, key_count)