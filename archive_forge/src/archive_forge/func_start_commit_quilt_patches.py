import shutil
import tempfile
from ... import errors
from ... import merge as _mod_merge
from ... import trace
from ...i18n import gettext
from ...mutabletree import MutableTree
from ...revisiontree import RevisionTree
from .quilt import QuiltPatches
def start_commit_quilt_patches(tree, policy):
    quilt = QuiltPatches.find(tree)
    if quilt is None:
        return
    applied_patches = quilt.applied()
    unapplied_patches = quilt.unapplied()
    if policy is None:
        if applied_patches and unapplied_patches:
            trace.warning(gettext('Committing with %d patches applied and %d patches unapplied.'), len(applied_patches), len(unapplied_patches))
    elif policy == 'applied':
        quilt.push_all()
    elif policy == 'unapplied':
        quilt.pop_all()