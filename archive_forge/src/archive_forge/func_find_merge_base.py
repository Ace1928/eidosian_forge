from heapq import heappop, heappush
from .lru_cache import LRUCache
def find_merge_base(repo, commit_ids):
    """Find lowest common ancestors of commit_ids[0] and *any* of commits_ids[1:].

    Args:
      repo: Repository object
      commit_ids: list of commit ids
    Returns:
      list of lowest common ancestor commit_ids
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
    if not commit_ids:
        return []
    c1 = commit_ids[0]
    if not len(commit_ids) > 1:
        return [c1]
    c2s = commit_ids[1:]
    if c1 in c2s:
        return [c1]
    lcas = _find_lcas(lookup_parents, c1, c2s, lookup_stamp)
    return lcas