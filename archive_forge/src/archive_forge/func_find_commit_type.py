from typing import FrozenSet, Optional, Set, Tuple
import gevent
from gevent import pool
from .object_store import (
from .objects import Commit, ObjectID, Tag
def find_commit_type(sha):
    try:
        o = obj_store[sha]
    except KeyError:
        if not ignore_unknown:
            raise
    else:
        if isinstance(o, Commit):
            commits.add(sha)
        elif isinstance(o, Tag):
            tags.add(sha)
            commits.add(o.object[1])
        else:
            raise KeyError('Not a commit or a tag: %s' % sha)