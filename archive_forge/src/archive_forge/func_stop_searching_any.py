import time
from . import debug, errors, osutils, revision, trace
def stop_searching_any(self, revisions):
    """
        Remove any of the specified revisions from the search list.

        None of the specified revisions are required to be present in the
        search list.

        It is okay to call stop_searching_any() for revisions which were seen
        in previous iterations. It is the callers responsibility to call
        find_seen_ancestors() to make sure that current search tips that are
        ancestors of those revisions are also stopped.  All explicitly stopped
        revisions will be excluded from the search result's get_keys(), though.
        """
    revisions = frozenset(revisions)
    if self._returning == 'next':
        stopped = self._next_query.intersection(revisions)
        self._next_query = self._next_query.difference(revisions)
    else:
        stopped_present = self._current_present.intersection(revisions)
        stopped = stopped_present.union(self._current_ghosts.intersection(revisions))
        self._current_present.difference_update(stopped)
        self._current_ghosts.difference_update(stopped)
        stop_rev_references = {}
        for rev in stopped_present:
            for parent_id in self._current_parents[rev]:
                if parent_id not in stop_rev_references:
                    stop_rev_references[parent_id] = 0
                stop_rev_references[parent_id] += 1
        for parents in self._current_parents.values():
            for parent_id in parents:
                try:
                    stop_rev_references[parent_id] -= 1
                except KeyError:
                    pass
        stop_parents = set()
        for rev_id, refs in stop_rev_references.items():
            if refs == 0:
                stop_parents.add(rev_id)
        self._next_query.difference_update(stop_parents)
    self._stopped_keys.update(stopped)
    self._stopped_keys.update(revisions)
    return stopped