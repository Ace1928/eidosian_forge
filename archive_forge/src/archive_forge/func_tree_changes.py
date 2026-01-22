import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def tree_changes(store, tree1_id, tree2_id, want_unchanged=False, rename_detector=None, include_trees=False, change_type_same=False):
    """Find the differences between the contents of two trees.

    Args:
      store: An ObjectStore for looking up objects.
      tree1_id: The SHA of the source tree.
      tree2_id: The SHA of the target tree.
      want_unchanged: If True, include TreeChanges for unmodified entries
        as well.
      include_trees: Whether to include trees
      rename_detector: RenameDetector object for detecting renames.
      change_type_same: Whether to report change types in the same
        entry or as delete+add.

    Returns:
      Iterator over TreeChange instances for each change between the
        source and target tree.
    """
    if rename_detector is not None and tree1_id is not None and (tree2_id is not None):
        yield from rename_detector.changes_with_renames(tree1_id, tree2_id, want_unchanged=want_unchanged, include_trees=include_trees)
        return
    entries = walk_trees(store, tree1_id, tree2_id, prune_identical=not want_unchanged)
    for entry1, entry2 in entries:
        if entry1 == entry2 and (not want_unchanged):
            continue
        entry1 = _skip_tree(entry1, include_trees)
        entry2 = _skip_tree(entry2, include_trees)
        if entry1 != _NULL_ENTRY and entry2 != _NULL_ENTRY:
            if stat.S_IFMT(entry1.mode) != stat.S_IFMT(entry2.mode) and (not change_type_same):
                yield TreeChange.delete(entry1)
                entry1 = _NULL_ENTRY
                change_type = CHANGE_ADD
            elif entry1 == entry2:
                change_type = CHANGE_UNCHANGED
            else:
                change_type = CHANGE_MODIFY
        elif entry1 != _NULL_ENTRY:
            change_type = CHANGE_DELETE
        elif entry2 != _NULL_ENTRY:
            change_type = CHANGE_ADD
        else:
            continue
        yield TreeChange(change_type, entry1, entry2)