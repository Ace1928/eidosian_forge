import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def walk_trees(store, tree1_id, tree2_id, prune_identical=False):
    """Recursively walk all the entries of two trees.

    Iteration is depth-first pre-order, as in e.g. os.walk.

    Args:
      store: An ObjectStore for looking up objects.
      tree1_id: The SHA of the first Tree object to iterate, or None.
      tree2_id: The SHA of the second Tree object to iterate, or None.
      prune_identical: If True, identical subtrees will not be walked.

    Returns:
      Iterator over Pairs of TreeEntry objects for each pair of entries
        in the trees and their subtrees recursively. If an entry exists in one
        tree but not the other, the other entry will have all attributes set
        to None. If neither entry's path is None, they are guaranteed to
        match.
    """
    mode1 = tree1_id and stat.S_IFDIR or None
    mode2 = tree2_id and stat.S_IFDIR or None
    todo = [(TreeEntry(b'', mode1, tree1_id), TreeEntry(b'', mode2, tree2_id))]
    while todo:
        entry1, entry2 = todo.pop()
        is_tree1 = _is_tree(entry1)
        is_tree2 = _is_tree(entry2)
        if prune_identical and is_tree1 and is_tree2 and (entry1 == entry2):
            continue
        tree1 = is_tree1 and store[entry1.sha] or None
        tree2 = is_tree2 and store[entry2.sha] or None
        path = entry1.path or entry2.path
        todo.extend(reversed(_merge_entries(path, tree1, tree2)))
        yield (entry1, entry2)