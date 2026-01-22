import collections
from .objects import ZERO_SHA, format_timezone, parse_timezone
Drop the specified reflog entry.

    Args:
        f: File-like object
        index: Reflog entry index (in Git reflog reverse 0-indexed order)
        rewrite: If a reflog entry's predecessor is removed, set its
            old SHA to the new SHA of the entry that now precedes it
    