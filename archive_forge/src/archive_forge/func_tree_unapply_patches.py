import shutil
import tempfile
from ... import errors
from ... import merge as _mod_merge
from ... import trace
from ...i18n import gettext
from ...mutabletree import MutableTree
from ...revisiontree import RevisionTree
from .quilt import QuiltPatches
def tree_unapply_patches(orig_tree, orig_branch=None, force=False):
    """Return a tree with patches unapplied.

    :param orig_tree: Tree from which to unapply quilt patches
    :param orig_branch: Related branch (optional)
    :return: Tuple with tree and temp path.
        The tree is a tree with unapplied patches; either a checkout of
        tree or tree itself if there were no patches
    """
    if orig_branch is None:
        orig_branch = orig_tree.branch
    quilt = QuiltPatches.find(orig_tree)
    if quilt is None:
        return (orig_tree, None)
    applied_patches = quilt.applied()
    if not applied_patches:
        return (orig_tree, None)
    target_dir = tempfile.mkdtemp()
    try:
        if isinstance(orig_tree, MutableTree):
            tree = orig_branch.create_checkout(target_dir, lightweight=True, revision_id=orig_tree.last_revision(), accelerator_tree=orig_tree)
            merger = _mod_merge.Merger.from_uncommitted(tree, orig_tree)
            merger.merge_type = NoUnapplyingMerger
            merger.do_merge()
        elif isinstance(orig_tree, RevisionTree):
            tree = orig_branch.create_checkout(target_dir, lightweight=True, accelerator_tree=orig_tree, revision_id=orig_tree.get_revision_id())
        else:
            trace.mutter('Not sure how to create copy of %r', orig_tree)
            shutil.rmtree(target_dir)
            return (orig_tree, None)
        trace.mutter('Applying quilt patches for %r in %s', orig_tree, target_dir)
        quilt = QuiltPatches.find(tree)
        if quilt is not None:
            quilt.pop_all(force=force)
        return (tree, target_dir)
    except BaseException:
        shutil.rmtree(target_dir)
        raise