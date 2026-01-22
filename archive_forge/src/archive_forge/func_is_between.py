import time
from . import debug, errors, osutils, revision, trace
def is_between(self, revid, lower_bound_revid, upper_bound_revid):
    """Determine whether a revision is between two others.

        returns true if and only if:
        lower_bound_revid <= revid <= upper_bound_revid
        """
    return (upper_bound_revid is None or self.is_ancestor(revid, upper_bound_revid)) and (lower_bound_revid is None or self.is_ancestor(lower_bound_revid, revid))