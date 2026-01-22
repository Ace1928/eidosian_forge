import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def build_commit_graph(object_store, commit_spec, trees=None, attrs=None):
    """Build a commit graph from a concise specification.

    Sample usage:
    >>> c1, c2, c3 = build_commit_graph(store, [[1], [2, 1], [3, 1, 2]])
    >>> store[store[c3].parents[0]] == c1
    True
    >>> store[store[c3].parents[1]] == c2
    True

    If not otherwise specified, commits will refer to the empty tree and have
    commit times increasing in the same order as the commit spec.

    Args:
      object_store: An ObjectStore to commit objects to.
      commit_spec: An iterable of iterables of ints defining the commit
        graph. Each entry defines one commit, and entries must be in
        topological order. The first element of each entry is a commit number,
        and the remaining elements are its parents. The commit numbers are only
        meaningful for the call to make_commits; since real commit objects are
        created, they will get created with real, opaque SHAs.
      trees: An optional dict of commit number -> tree spec for building
        trees for commits. The tree spec is an iterable of (path, blob, mode)
        or (path, blob) entries; if mode is omitted, it defaults to the normal
        file mode (0100644).
      attrs: A dict of commit number -> (dict of attribute -> value) for
        assigning additional values to the commits.
    Returns: The list of commit objects created.

    Raises:
      ValueError: If an undefined commit identifier is listed as a parent.
    """
    if trees is None:
        trees = {}
    if attrs is None:
        attrs = {}
    commit_time = 0
    nums = {}
    commits = []
    for commit in commit_spec:
        commit_num = commit[0]
        try:
            parent_ids = [nums[pn] for pn in commit[1:]]
        except KeyError as exc:
            missing_parent, = exc.args
            raise ValueError('Unknown parent %i' % missing_parent) from exc
        blobs = []
        for entry in trees.get(commit_num, []):
            if len(entry) == 2:
                path, blob = entry
                entry = (path, blob, F)
            path, blob, mode = entry
            blobs.append((path, blob.id, mode))
            object_store.add_object(blob)
        tree_id = commit_tree(object_store, blobs)
        commit_attrs = {'message': ('Commit %i' % commit_num).encode('ascii'), 'parents': parent_ids, 'tree': tree_id, 'commit_time': commit_time}
        commit_attrs.update(attrs.get(commit_num, {}))
        commit_obj = make_commit(**commit_attrs)
        commit_time = commit_attrs['commit_time'] + 100
        nums[commit_num] = commit_obj.id
        object_store.add_object(commit_obj)
        commits.append(commit_obj)
    return commits