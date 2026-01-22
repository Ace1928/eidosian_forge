import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def tree_changes_for_merge(store, parent_tree_ids, tree_id, rename_detector=None):
    """Get the tree changes for a merge tree relative to all its parents.

    Args:
      store: An ObjectStore for looking up objects.
      parent_tree_ids: An iterable of the SHAs of the parent trees.
      tree_id: The SHA of the merge tree.
      rename_detector: RenameDetector object for detecting renames.

    Returns:
      Iterator over lists of TreeChange objects, one per conflicted path
      in the merge.

      Each list contains one element per parent, with the TreeChange for that
      path relative to that parent. An element may be None if it never
      existed in one parent and was deleted in two others.

      A path is only included in the output if it is a conflict, i.e. its SHA
      in the merge tree is not found in any of the parents, or in the case of
      deletes, if not all of the old SHAs match.
    """
    all_parent_changes = [tree_changes(store, t, tree_id, rename_detector=rename_detector) for t in parent_tree_ids]
    num_parents = len(parent_tree_ids)
    changes_by_path: Dict[str, List[Optional[TreeChange]]] = defaultdict(lambda: [None] * num_parents)
    for i, parent_changes in enumerate(all_parent_changes):
        for change in parent_changes:
            if change.type == CHANGE_DELETE:
                path = change.old.path
            else:
                path = change.new.path
            changes_by_path[path][i] = change

    def old_sha(c):
        return c.old.sha

    def change_type(c):
        return c.type
    for _, changes in sorted(changes_by_path.items()):
        assert len(changes) == num_parents
        have = [c for c in changes if c is not None]
        if _all_eq(have, change_type, CHANGE_DELETE):
            if not _all_same(have, old_sha):
                yield changes
        elif not _all_same(have, change_type):
            yield changes
        elif None not in changes:
            yield changes