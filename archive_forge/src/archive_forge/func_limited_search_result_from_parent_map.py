import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def limited_search_result_from_parent_map(parent_map, missing_keys, tip_keys, depth):
    """Transform a parent_map that is searching 'tip_keys' into an
    approximate SearchResult.

    We should be able to generate a SearchResult from a given set of starting
    keys, that covers a subset of parent_map that has the last step pointing at
    tip_keys. This is to handle the case that really-long-searches shouldn't be
    started from scratch on each get_parent_map request, but we *do* want to
    filter out some of the keys that we've already seen, so we don't get
    information that we already know about on every request.

    The server will validate the search (that starting at start_keys and
    stopping at stop_keys yields the exact key_count), so we have to be careful
    to give an exact recipe.

    Basic algorithm is:
        1) Invert parent_map to get child_map (todo: have it cached and pass it
           in)
        2) Starting at tip_keys, walk towards children for 'depth' steps.
        3) At that point, we have the 'start' keys.
        4) Start walking parent_map from 'start' keys, counting how many keys
           are seen, and generating stop_keys for anything that would walk
           outside of the parent_map.

    :param parent_map: A map from {child_id: (parent_ids,)}
    :param missing_keys: parent_ids that we know are unavailable
    :param tip_keys: the revision_ids that we are searching
    :param depth: How far back to walk.
    """
    if not parent_map:
        return ([], [], 0)
    heads = _find_possible_heads(parent_map, tip_keys, depth)
    s, found_heads = _run_search(parent_map, heads, set(tip_keys))
    start_keys, exclude_keys, keys = s.get_state()
    if found_heads:
        start_keys = set(start_keys).difference(found_heads)
    return (start_keys, exclude_keys, len(keys))