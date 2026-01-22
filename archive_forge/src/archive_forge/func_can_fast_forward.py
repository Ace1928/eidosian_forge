from heapq import heappop, heappush
from .lru_cache import LRUCache
def can_fast_forward(repo, c1, c2):
    """Is it possible to fast-forward from c1 to c2?

    Args:
      repo: Repository to retrieve objects from
      c1: Commit id for first commit
      c2: Commit id for second commit
    """
    cmtcache = LRUCache(max_cache=128)
    parents_provider = repo.parents_provider()

    def lookup_stamp(cmtid):
        if cmtid not in cmtcache:
            cmtcache[cmtid] = repo.object_store[cmtid]
        return cmtcache[cmtid].commit_time

    def lookup_parents(cmtid):
        commit = None
        if cmtid in cmtcache:
            commit = cmtcache[cmtid]
        return parents_provider.get_parents(cmtid, commit=commit)
    if c1 == c2:
        return True
    min_stamp = lookup_stamp(c1)
    lcas = _find_lcas(lookup_parents, c1, [c2], lookup_stamp, min_stamp=min_stamp)
    return lcas == [c1]